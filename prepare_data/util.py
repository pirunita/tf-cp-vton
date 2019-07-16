
from scipy.misc import imresize

import numpy as np
import tensorflow as tf

class ImageDecoder(object):
    """Helper class for decoding images in Tensorflow"""

    def __init__(self):
        # Create a single Tensorflow Session for all image decoding calls.
        self._sess = tf.Session()

        # Tensorflow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)
        self._decoded_jpeg = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
        self._encode_jpeg = tf.image.encode_jpeg(self._decode_jpeg)
        
        # Tensorflow ops for PNG decoding
        self._encoded_png = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._encoded_png, channels=3)
        self._decoded_png = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
        self._encode_png = tf.image.encode_png(self._decoded_png)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                            feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def decode_png(self, encoded_png):
        image = self._sess.run(self._decode_png,
                            feed_dict={self._encoded_png: encoded_png})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, decoded_image):
        image = self._sess.run(self._encode_jpeg,
                            feed_dict={self._decoded_jpeg: decoded_image})
        return image
    
    def encode_png(self, decoded_image):
        image = self._sess.run(self._encode_png,
                            feed_dict={self._decoded_png: decoded_image})
        return image


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def process_pose_map(pose_keypoints, h ,w):
    """Given 18 * 2 keypoints, return resize_h * resize_w * 18 map"""
    resize_w = 192.0
    resize_h = 256.0
    pose_keypoints = np.asarray(pose_keypoints, np.float32)
    pose_keypoints[:, 0] = pose_keypoints[:, 0] * resize_w / float(w)
    pose_keypoints[:, 1] = pose_keypoints[:, 1] * resize_h / float(h)
    pose_keypoints = np.asarray(pose_keypoints, np.int)

    pose_map = np.zeros((int(resize_h),int(resize_w),18), np.bool)
    for i in range(18):
        if pose_keypoints[i,0] < 0:
            continue
        t = np.max((pose_keypoints[i,1] - 5, 0))
        b = np.min((pose_keypoints[i,1] + 5, h - 1))
        l = np.max((pose_keypoints[i,0] - 5, 0))
        r = np.min((pose_keypoints[i,0] + 5, w - 1))
        pose_map[t:b+1, l:r+1, i] = True

    return pose_map.tostring()
def process_segment_map(segment, h, w):
    """Extract segment maps. """
    segment = np.asarray(segment, dtype=np.uint8)
    segment = imresize(segment, (h, w), interp='nearest')
    return segment.tostring()
    