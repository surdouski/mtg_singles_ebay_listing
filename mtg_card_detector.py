import argparse
import os

import cv2
import pandas as pd
import numpy as np
import ntpath

from settings import PICKLE_PATH
from fetch_data import fetch_cards
from hashing import fetch_cards_pool_with_hashed_images
from opencv_dnn import detect_frame


def run_from_command_line(command_line_args):
    if _invalid_path(command_line_args.in_path):
        return _invalid_path_message(os.path.abspath(command_line_args.in_path))
    out_path = _parse_path(command_line_args)

    if _invalid_path(command_line_args.pickle_path):
        cards_df = fetch_cards()
        fetch_cards_pool_with_hashed_images(cards_df, save_to=command_line_args.pickle_path)

    card_pool = pd.read_pickle(command_line_args.pickle_path)
    card_hk = f'card_hash_{command_line_args.hash_size}'
    card_pool = card_pool[
        ['name', 'set', 'collector_number', card_hk, 'prices']
    ]
    card_pool[card_hk] = card_pool[card_hk].apply(lambda x: x.hash.flatten())

    img = cv2.imread(command_line_args.in_path)
    detected_cards, image_result = detect_frame(
        img,
        card_pool,
        hash_size=command_line_args.hash_size,
    )
    if out_path is not None:
        cv2.imwrite(f'images/output/{ntpath.basename(command_line_args.in_path)}', image_result.astype(np.uint8))


def _parse_path(command_line_args):
    if not command_line_args.out_path:
        return
    f_name = os.path.split(command_line_args.in_path)[1]
    return '%s/%s.avi' % (command_line_args.out_path, f_name[:f_name.find('.')])


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in', dest='in_path', help='Path of the input file.', type=str)
    parser.add_argument('-o', '--out', dest='out_path', help='Path of the output directory to save the result',
                        type=str)
    parser.add_argument('-p', '--pickle', dest='pickle_path', help='Path of the pickled card image hash.', type=str,
                        default=PICKLE_PATH)
    parser.add_argument('-hs', '--hash_size', dest='hash_size', help='Size of the hash for pHash algorithm', type=int,
                        default=16)
    return parser


def _invalid_path(path):
    return not os.path.isfile(path)


def _invalid_path_message(path):
    print('Input file does not exist at path specified {path}')  # super serious error logging


if __name__ == '__main__':
    parser = create_parser()
    command_line_args = parser.parse_args()

    if command_line_args.in_path is None:
        print('No input file.')
        exit()
    if _invalid_path(command_line_args.in_path):
        print('invalid input file')
        exit()
    run_from_command_line(command_line_args)

