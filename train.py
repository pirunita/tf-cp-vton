import argparse
import os
import time

import tensorflow as tf

from gmm_dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(description='Training codes for GMM using Tensorflow')
    parser.add_argument('--model', default='GMM')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--values_per_shards', type=int, help='Change based # of dataset')
    parser.add_argument('--num_preprocess_threads', type=int, default=8)

    parser.add_argument('--fine_width', type=int, default=192)
    parser.add_argument('--fine_height', type=int, default=256)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    parser.add_argument('--data_mode', default='train')
    parser.add_argument('--input_file_pattern', default='./prepare_data/tfrecord/train-?????-of-00032')
    parser.add_argument('--data_list', default='train_pairs.txt')
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    args = parser.parse_args()

    print(args)

    # create dataset
    train_dataset = build_dataset(args)



if __name__ == "__main__":
    main()