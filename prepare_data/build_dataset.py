from collections import namedtuple
from datetime import datetime

import argparse
import os
import random
import sys
import threading

import numpy as np
import scipy.io as sio
import tensorflow as tf

from util import _bytes_feature
from util import _int64_feature
from util import ImageDecoder
from util import process_pose_map
from util import process_segment_map

def build_tf_example(input_dir, image, decoder):
    """Builds a TF Example proto for an image pair, pose and segmentation
    
    Args:
        image: An ImageMetadata object.
        decoder: an ImageDecoder object.
    
    Returns:
        A TF Example proto
    """
    pose_dir = os.path.join(input_dir, 'pose')
    segment_dir = os.path.join(input_dir, 'segment')
    with open(input_dir + '/image/' + image.image_id, 'r') as f1:
        encoded_image = f1.read()
    with open(input_dir + '/product_image/' + image.product_image_id, 'r') as f2:
        encoded_prod_image = f2.read()
    with open(input_dir + '/product_mask/' + image.product_image_id, 'r') as f3:
        encoded_prod_mask = f3.read()

    # Check image format
    if os.path.splitext(image.image_id)[1][1:] == 'png':
        try:
            decoded_image = decoder.decode_png(encoded_image)
            decoder.decode_png(encoded_prod_image)
            decoder.decode_png(encoded_prod_mask)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid PNG data: %s" % image.image_id)
            print("Skipping file with invalid PNG data: %s" % image.product_image_id)
            return

    elif os.path.splitext(image.image_id)[1][1:] == 'jpg':
        try:
            decoded_image = decoder.decode_jpeg(encoded_image)
            decoder.decode_jpeg(encoded_prod_image)
            decoder.decode_jpeg(encoded_prod_mask)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % image.image_id)
            print("Skipping file with invalid JPEG data: %s" % image.product_image_id)
            return

    height = decoded_image.shape[0]
    width = decoded_image.shape[1]
    pose_map = sio.loadmat(os.path.join(pose_dir, image.image_id[:-4] + '.mat'))
    pose_map_str = process_pose_map(pose_map, height, width)
    segment_map = sio.loadmat(os.path.join(segment_dir, image.image_id[:-4] + '.mat'))
    segment_map_str = process_segment_map(segment_map['segment'], height, width)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image_id': _bytes_feature(image.image_id),
        'product_image_id': _bytes_feature(image.product),
        'image': _bytes_feature(encoded_image),
        'product_image': _bytes_feature(encoded_prod_image),
        'product_mask': _bytes_feature(encoded_prod_mask),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'pose_map': _bytes_feature(pose_map_str),
        'segment_map': _bytes_feature(segment_map_str)
    }))

    return tf_example


def load_metadata(args):
    """Loads image metadata from a text file

    Args:
        args: parser
    Returns:
        A list of ImageMeadata
    """
    image_pairs = open(os.path.join(args.data_root, args.data_list)).read().splitlines()

    ImageMetaData = namedtuple('ImageMetadata',
                                ['image_id', 'product_image_id'])
    image_metadata = []

    for item in image_pairs:
        image_pair = item.split()
        image_metadata.append(ImageMetaData(image_pair[0], image_pair[1]))
    
    print('Finished processing %d pairs for %d images in %s' %
          (len(image_pairs), len(image_pairs), args.data_list))

    return image_metadata
    

def process_dataset(args, image_metadata, prefix='train'):
    """save data as a TFRecord
    """

    # Shuffle the ordering of images
    random.seed(12345)
    random.shuffle(image_metadata)

    # Batch i is defined as image_metadata[ranges[i][0]:rangies[i][1]]
    num_threads = min(args.train_shards, args.num_threads)
    spacing = np.linspace(0, len(image_metadata), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])
    
    coord = tf.train.Coordinator()
    decoder = ImageDecoder()

    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        argument = (args.data_root, args.output_dir, thread_index, 
                    ranges, prefix, image_metadata, decoder, args.train_shards)
        t = threading.Thread(target=process_files, args=argument)
        t.start()
        threads.append(t)
    
    coord.join(threads)
    print("%s: Finished processing all %d image pairs in data set '%s'." %
        (datetime.now(), len(image_metadata), prefix))

def process_files(input_dir, output_dir,
                  thread_index, ranges, prefix, 
                  image_metadata, decoder,
                  train_shards):
    """Process and save a subset of images as TFRecord files in one thread

    Args:
        input_dir: data_root
        output_dir: ./tfrecord/
        thread_index: Integer thread identifier within [0, len(ranges)].
        ranges: A list of pairs of integers specifying the ranges of the dataset to
                process in parallel.
        prefix: specifying the dataset.
        image_metadata: list of ImageMetadata
        decoder: An ImageDecoder object
        num_shards: Integer number of shards for the output files.
    """

    num_threads = len(ranges)
    assert not train_shards % num_threads
    num_shards_per_batch = int(train_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch+1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for i in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + i
        output_filename = '%s-%.5d-of-%.5d' % (prefix, shard, train_shards)
        output_file = os.path.join(output_dir, output_filename)
        tfrecord_writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[i], shard_ranges[i+1], dtype=int)
        for j in images_in_shard:
            image = image_metadata[j]
            tf_example = build_tf_example(input_dir, image, decoder)
            
            if tf_example is not None:
                tfrecord_writer.write(tf_example.SerializeToString())
                shard_counter += 1
                counter += 1
            
            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                    (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        tfrecord_writer.close()
        print("%s [thread %d]: Wrote %d image pairs to %s" %
            (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='Build tfrecord for GMM using tensorflow')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--data_mode', default='train')
    parser.add_argument('--data_list', default='train_pairs.txt')
    
    parser.add_argument('--output_dir', default='./tfrecord/')

    parser.add_argument('--train_shards', type=int, default=32)
    parser.add_argument('--num_threads', type=int, default=8)

    args = parser.parse_args()

    if not tf.gfile.IsDirectory(args.output_dir):
        tf.gfile.MakeDirs(args.output_dir)

        image_metadata = load_metadata(args)
        process_dataset(args, image_metadata)


if __name__ == "__main__":
    main()