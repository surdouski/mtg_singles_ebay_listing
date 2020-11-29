import ast

import cv2
import imagehash as ih
import numpy as np
import pandas as pd
from PIL import Image


from config import Config
from fetch_data import get_valid_filename
from fetch_data import fetch_card_image as _fetch_card_image_from_web


def fetch_cards_pool_with_hashed_images(card_pool, save_to=None, _hash_size=None):
    """Calculate perceptual hash (pHash) value for each cards in the database,
        then store them if needed

    Parameters
    ----------
    card_pool : pd.DataFrame
    save_to : str
        (default is None)
    _hash_size : int

    Returns
    -------
    pd.DataFrame
    """

    hash_sizes = [16, 32] if _hash_size is None else [_hash_size]
    cards_pool = create_empty_cards_pool(card_pool, hash_sizes)

    for ind, card_info in card_pool.iterrows():
        print_update_message(ind)  # print update message every modulus 100 to keep tabs on hashing
        card_names = fetch_card_names_from_database(card_info)  # fetch from local csv files
        update_card_info_for_names(card_info, card_names, cards_pool, hash_sizes)  # try local, then web
    save_cards_pool_to_pickle(cards_pool, save_to)
    return cards_pool


def update_card_info_for_names(card_info, card_names, cards_pool, hash_sizes):
    for card_name in card_names:
        update_card_info(cards_pool, card_info, card_name, hash_sizes)


def save_cards_pool_to_pickle(cards_pool, save_to):
    if save_to is not None:
        cards_pool.to_pickle(save_to)


def update_card_info(alternate_card_format_pool, card_info, card_name, hash_sizes):
    update_card_info_name(card_info, card_name)
    card_image = fetch_card_image(card_info)
    if card_image:
        update_card_info_hashes(card_image, card_info, hash_sizes)
        update_alternate_card_format_pool(alternate_card_format_pool, card_info)
    else:
        display_warning(card_info)


def update_card_info_name(card_info, card_name):
    card_info['name'] = card_name


def update_card_info_hashes(card_image, card_info, hash_sizes):
    for hash_size in hash_sizes:
        card_info[f'card_hash_{hash_size}'] = ih.phash(
            Image.fromarray(card_image),
            hash_size=hash_size
        )


def display_warning(card_info):
    print(f"WARNING: card {card_image_path(card_info)} is not found!")


def fetch_card_image(card_info):
    card_image = fetch_card_image_from_database(card_info)
    if not card_image:
        card_image = fetch_card_image_from_web(card_image, card_info)
    return card_image


def update_alternate_card_format_pool(alternate_card_format_pool, card_info):
    alternate_card_format_pool.loc[
        0 if alternate_card_format_pool.empty else alternate_card_format_pool.index.max() + 1] = card_info


def create_empty_cards_pool(card_pool, hash_sizes):
    alternate_card_format_pool = pd.DataFrame(columns=list(card_pool.columns.values))
    for hash_size in hash_sizes:
        alternate_card_format_pool[f'card_hash_{hash_size}'] = np.NaN
    return alternate_card_format_pool


def fetch_card_image_from_web(card_image, card_info):
    _fetch_card_image_from_web(
        card_info,
        out_dir=f"{Config.data_dir}/card_img/png/{card_info['set']}"
    )
    return cv2.imread(card_image_path(card_info))


def fetch_card_image_from_database(card_info):
    return cv2.imread(card_image_path(card_info))


def card_image_path(card_info):
    return f"{Config.data_dir}/card_img/png/{card_info['set']}/" \
                 f"{card_info['collector_number']}_{get_valid_filename(card_info['name'])}"


def fetch_card_names_from_database(card_info):
    card_names = []
    if _card_format_is_double_faced_token(card_info):
        return append_card_faces_to_names(card_info)
    else:
        card_names.append(card_info['name'])
    return card_names


def _card_format_is_double_faced_token(card_info):
    return card_info['layout'] in ['transform', 'double_faced_token']


def append_card_faces_to_names(card_info):
    card_faces = fetch_card_faces(card_info)
    return [card_face['name'] for card_face in card_faces]


def fetch_card_faces(card_info):
    if isinstance(card_info['card_faces'], str):
        card_faces = ast.literal_eval(card_info['card_faces'])
    else:
        card_faces = card_info['card_faces']
    return card_faces


def print_update_message(ind):
    if ind % 100 == 0:
        print('Calculating hashes: %dth card' % ind)
