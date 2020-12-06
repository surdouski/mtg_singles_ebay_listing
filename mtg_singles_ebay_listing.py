import argparse
import os

import cv2
import pandas as pd
import tkinter as tk

from settings import PROJECT_ROOT, PICKLE_PATH, IMAGE_PATH

from src.graphical_ui import ImageDetailsConfirmation
from src.fetch_data import fetch_cards
from src.hashing import fetch_cards_pool_with_hashed_images
from src.image_matcher import perform_card_detection


def main():
    card_pool = retrieve_from_database_and_flatten_results()
    input_image = cv2.imread(input_path)
    ebay_listing_objects = perform_card_detection(input_image, input_path, output_path, card_pool, hash_size=hash_size)
    send_to_interface_for_validation(ebay_listing_objects)


def retrieve_from_database_and_flatten_results():
    card_pool = get_valid_pickle_or_create_new(command_line_args.pickle_path)
    card_pool = flatten_hash_array(card_pool)
    return card_pool


def send_to_interface_for_validation(ebay_listing_objects):
    for card in ebay_listing_objects:
        confirm_or_reject_listing(card)


def confirm_or_reject_listing(card):
    root = tk.Tk()
    root.geometry("300x200+300+300")
    app = ImageDetailsConfirmation(card)
    root.mainloop()


def flatten_hash_array(card_pool):
    card_hk = f'card_hash_{hash_size}'
    print(card_hk)
    card_pool = card_pool[
        ['id', 'name', 'set', 'collector_number', card_hk]
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


def _get_valid_path_or_exit(path, path_type):
    if not path:
        print(f'No path set for {path_type}.')
        exit()
    if _invalid_path(path):
        print(f'Invalid path for {path_type}.')
        exit()
    return path


def _get_valid_dir_or_exit(_dir, dir_path):
    if not _dir:
        print(f'No dir set for {dir_path}.')
        exit()
    if _invalid_dir(_dir):
        print(f'Invalid dir for {dir_path}.')
        exit()
    return _dir


def _get_valid_hash_size_or_exit(_hash_size):
    if _hash_size not in [16, 32]:
        print('Hash size must be 16 or 32.')
        exit()
    return _hash_size


def _invalid_path(path):
    return not os.path.isfile(path)


def _invalid_dir(_dir):
    return not os.path.isdir(_dir)


if __name__ == '__main__':
    parser = create_parser()
    command_line_args = parser.parse_args()

    input_path = _get_valid_path_or_exit(command_line_args.in_path, 'input path')
    output_path = _get_valid_dir_or_exit(command_line_args.out_path, 'output path')
    pickle_path = command_line_args.pickle_path
    hash_size = _get_valid_hash_size_or_exit(command_line_args.hash_size)

    main()
