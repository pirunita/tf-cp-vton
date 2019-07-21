import os

import tensorflow as tf

from util import read_imgfile


def build_dataset(args):
    """Load input data from Tfrecord"""
    input_queue = prefetch_input_dataset(
          tf.TFRecordReader(),
          args.input_file_pattern,
          args.batch_size,
          args.values_per_shards,
          input_queue_capacity_factor=2,
          num_reader_threads=args.num_preprocess_threads)

    images_and_maps = []
    
    for thread_id in range(args.num_preprocess_threads):
        serialized_dataset = input_queue.dequeue()
        (encoded_image, encoded_product_image, encoded_product_mask, 
         body_segment, prod_segment, skin_segment, pose_map, 
         image_id) = parsing_tfrecord(serialized_dataset)

        (image, product_image, product_mask, body_segment,
         prod_segment, skin_segment, pose_map) = process_image(
                                                encoded_image,
                                                encoded_product_image,
                                                encoded_product_mask,
                                                body_segment,
                                                prod_segment,
                                                skin_segment,
                                                pose_map)
        
        images_and_maps.append([image, product_image, body_segment, 
                                prod_segment, skin_segment, pose_map, image_id])
    
    # Batch
    """ 7 * 8 * 4= 224
    """
    queue_capacity = (7 * args.num_preprocess_threads * args.batch_size)

    return tf.train.batch_join(images_and_maps,
                               batch_size=args.batch_size,
                               capacity=queue_capacity,
                               name='batch')



def extract_segmentation(segment):
    """Given semantic segmentation map, extract the body part."""
    """
        # 0 : 배경
        # 1 : 모자, 2 : 머리, 3 : 장갑, 4 : 선글라스, 5 : 상의
        # 6 : 드레스, 7 : 코트, 8 : 양말, 9 : 바지, 10 : 원피스형 수트
        # 11 : 스카프, 12 : 치마, 13 : 얼굴, 14 : 왼쪽 팔, 15 : 오른팔
        # 16 : 왼쪽 다리, 17 : 오른쪽 다리, 18 : 왼쪽 신발, 18 : 오른쪽 신발
    """
    product_segmentation = tf.cast(tf.equal(segment, 5), tf.float32)


    skin_segmentation = (tf.cast(tf.equal(segment, 1), tf.float32) +
                        tf.cast(tf.equal(segment, 2), tf.float32) +
                        tf.cast(tf.equal(segment, 4), tf.float32) +
                        tf.cast(tf.equal(segment, 13), tf.float32))

    body_segmentation = (1.0 - tf.cast(tf.equal(segment, 0), tf.float32) -
                            skin_segmentation)


    # Extend the axis
    product_segmentation = tf.expand_dims(product_segmentation, -1)
    body_segmentation = tf.expand_dims(body_segmentation, -1)
    skin_segmentation = tf.expand_dims(skin_segmentation, -1)

    body_segmentation = tf.image.resize_images(body_segmentation,
                                    size=[16, 12],
                                    method=tf.image.ResizeMethod.AREA,
                                    align_corners=False)

    return body_segmentation, product_segmentation, skin_segmentation


def parsing_tfrecord(serialized, stage=""):
    """parsing serialzied dataset into an image and caption"""
    features = tf.parse_single_example(
        serialized,
        features={
            'image_id': tf.FixedLenFeature([], tf.string),
            'product_image_id': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'product_image': tf.FixedLenFeature([], tf.string),
            'product_mask': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'pose_map': tf.FixedLenFeature([], tf.string),
            'segment_map': tf.FixedLenFeature([], tf.string)
        }
    )
    encoded_image = features['image']
    encoded_product_image = features['product_image']
    encoded_product_mask = features['product_mask']

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    pose_map = tf.decode_raw(features['pose_map'], tf.uint8)
    pose_map = tf.cast(pose_map, tf.float32)
    pose_map = tf.reshape(pose_map, tf.stack([256, 192, 18]))

    segment_map = tf.decode_raw(features['segment_map'], tf.uint8)
    segment_map = tf.reshape(segment_map, tf.stack([height, width]))
    body_segment, prod_segment, skin_segment = extract_segmentation(segment_map)

    return (encoded_image, encoded_product_image, encoded_product_mask,
            body_segment, prod_segment, skin_segment, pose_map, features['image_id'])


def prefetch_input_dataset(reader,
                           file_pattern,
                           batch_size,
                           values_per_shards,
                           input_queue_capacity_factor,
                           num_reader_threads,
                           shard_queue_name="filename queue",
                           value_queue_name="input_queue"):
    data_files = []

    for pattern in file_pattern.split('.'):
        data_files.extend(tf.gfile.Glob(pattern))
    
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)
    
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shards * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)

    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))
    tf.summary.scalar(
        "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity))
    
    return values_queue


def process_image(encoded_image,
                  encoded_product_image,
                  encoded_product_mask,
                  body_segment,
                  prod_segment,
                  skin_segment,
                  pose_map,
                  height=256,
                  width=192,
                  resize_height=256,
                  resize_width=192,
                  thread_id=0,
                  image_format='png',
                  zero_one_mask=True,
                  different_image_size=False):
    """Decode string to image, resize and apply random distortions.

    Args:
        encoded_image: String tensor containing image.
        encoded_product_image: String tensor containing product image.
        encoded_product_mask: String tensor containing product mask
        body_segment: Matrix containing segmentation of body.
        prod_segment: Matrix containing segmentation of product part
        skin_segment: Matrix containing segmentation of face part.
        pose_map: Matrix contaitning the pose keypoints.
        height: Height of output image.
        width: Width of output image.
        thread_id: Preprocessing thread id used to select the ordering of color distortions.
        zero_one_mask: True if use 0, 1 mask, false if use -1, 1 mask.
    
    """
    with tf.name_scope("decode", values=[encoded_image]):
        if image_format == 'jpeg':
            image = tf.image.decode_jpeg(encoded_image, channels=3)
            product_image = tf.image.decode_jpeg(encoded_product_image)
            product_mask = tf.image.decode_jpeg(encoded_product_mask, channels=1)

        elif image_format == 'png':
            image = tf.image.decode_png(encoded_image, channels=3)
            product_image = tf.image.decode_png(encoded_product_image)
            product_mask = tf.image.decode_png(encoded_product_mask, channels=1)
        
        else:
            raise ValueError("Invalid image format: %s" % image_format)
        
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        product_image = tf.image.convert_image_dtype(product_image, dtype=tf.float32)
        # pirunita, product_mask dtype값 cp-vton에서 다시 확인해야함.
        product_mask = tf.image.convert_image_dtype(product_mask, dtype=tf.float32)
    
    # Resize need?

    body_segment = tf.image.resize_images(body_segment,
                                size=[resize_height, resize_width],
                                      method=tf.image.ResizeMethod.BILINEAR,
                                      align_corners=False)
    skin_segment = tf.image.resize_images(skin_segment,
                                    size=[resize_height, resize_width],
                                        method=tf.image.ResizeMethod.BILINEAR,
                                        align_corners=False)
    
    prod_segment = tf.image.resize_images(prod_segment,
                                    size=[resize_height, resize_width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Rescale [0, 1] to [-1, 1] == transform
    image = (image - 0.5) * 2.0
    product_image = (product_image - 0.5) * 2.0
    
    skin_segment = skin_segment * image

    # zero_one mask need?

    return image, product_image, product_mask, body_segment, prod_segment, skin_segment, pose_map

class GmmDataset():
    """Dataset for GMM
    """
    def __init__(self, args):
        super(GmmDataset, self).__init__()
        self.args = args
        
        self.data_root = args.data_root
        self.data_list = args.data_list
        self.fine_height = args.fine_height
        self.fine_width = args.fine_width
        self.radius = args.radius
        self.data_path = os.path.join(args.data_root, args.data_mode)

        image_names = []
        cloth_image_names = []
        with open(os.path.join(args.data_root, args.data_list), 'r') as f:
            image_pairs = f.read().splitlines()
            for item in image_pairs:
                image_pair = item.split()
                image_names.append(image_pair[0])
                cloth_image_names.append(image_pair[1])

        self.image_names = image_names
        self.cloth_image_names = cloth_image_names
    
    def __getitem__(self, index):
        image_name = self.cloth_image_names[index]
        cloth_image_name = self.cloth_image_names[index]

        cloth

        