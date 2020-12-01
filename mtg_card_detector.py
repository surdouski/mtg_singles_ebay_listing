import argparse
import os

import cv2
import pandas as pd
import numpy as np
import ntpath

from settings import PICKLE_PATH, IMAGE_PATH
from fetch_data import fetch_cards
from hashing import fetch_cards_pool_with_hashed_images
from opencv_dnn import perform_card_detection


def run_from_command_line():
    card_pool = get_valid_pickle_or_create_new(command_line_args.pickle_path)
    card_pool = flatten_hash_array(card_pool)
    input_image = cv2.imread(input_path)
    perform_card_detection(input_image, input_path, output_path, card_pool, hash_size=hash_size)


def flatten_hash_array(card_pool):
    card_hk = f'card_hash_{hash_size}'
    card_pool = card_pool[
        ['name', 'set', 'collector_number', card_hk, 'prices']
    ]
    card_pool[card_hk] = card_pool[card_hk].apply(lambda x: x.hash.flatten())
    return card_pool


def get_valid_pickle_or_create_new(_pickle_path):
    if _invalid_path(_pickle_path):
        cards_df = fetch_cards()
        fetch_cards_pool_with_hashed_images(cards_df, save_to=_pickle_path)
    return pd.read_pickle(_pickle_path)


def _parse_path(command_line_args):
    if not command_line_args.out_path:
        return
    f_name = os.path.split(command_line_args.in_path)[1]
    return '%s/%s.avi' % (command_line_args.out_path, f_name[:f_name.find('.')])


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in', dest='in_path', help='Path of the input file.', type=str)
    parser.add_argument('-o', '--out', dest='out_path', help='Path of the output directory to save the result',
                        type=str, default=IMAGE_PATH)
    parser.add_argument('-p', '--pickle', dest='pickle_path', help='Path of the pickled card image hash.', type=str,
                        default=PICKLE_PATH)
    parser.add_argument('-hs', '--hash_size', dest='hash_size', help='Size of the hash for pHash algorithm', type=int,
                        default=16)
    return parser


def get_valid_path_or_exit(path, path_type):
    if not path:
        print(f'No path set for {path_type}.')
        exit()
    if _invalid_path(path):
        print(f'Invalid path for {path_type}.')
        exit()
    return path


def _invalid_path(path):
    return not os.path.isfile(path)


def get_valid_hash_size_or_exit(_hash_size):
    if _hash_size not in [16, 32]:
        print('Hash size must be 16 or 32.')
        exit()
    return _hash_size


if __name__ == '__main__':
    parser = create_parser()
    command_line_args = parser.parse_args()

    input_path = get_valid_path_or_exit(command_line_args.in_path, 'input path')
    output_path = get_valid_path_or_exit(command_line_args.out_path, 'output path')
    hash_size = get_valid_hash_size_or_exit(command_line_args.hash_size)

    run_from_command_line()

