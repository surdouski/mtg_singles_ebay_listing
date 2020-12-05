import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATA_DIR = os.getenv('DATA_DIR')

PICKLE_PATH = os.getenv('PICKLE_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')

PAYPAL_EMAIL = os.getenv('PAYPAL_EMAIL')

COUNTRY = os.getenv('COUNTRY')
POSTAL_CODE = os.getenv('POSTAL_CODE')
