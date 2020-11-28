import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

CARD_MASK_PATH = os.getenv('CARD_MASK_PATH')
DATA_DIR = os.getenv('DATA_DIR')
DARKNET_DIR = os.getenv('DARKNET_DIR')

PICKLE_PATH = os.getenv('PICKLE_PATH')
