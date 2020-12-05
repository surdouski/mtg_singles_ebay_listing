import os
from os.path import join, dirname
from dotenv import load_dotenv


PROJECT_ROOT = os.path.abspath(dirname(__file__))

DOTENV_PATH = join(PROJECT_ROOT, '.env')
print(DOTENV_PATH)
load_dotenv(DOTENV_PATH)

PICKLE_PATH = f"{PROJECT_ROOT}{os.getenv('PICKLE_PATH')}"
IMAGE_PATH = f"{PROJECT_ROOT}{os.getenv('IMAGE_PATH')}"

DATA_DIR = os.getenv('DATA_DIR')

PAYPAL_EMAIL = os.getenv('PAYPAL_EMAIL')
COUNTRY = os.getenv('COUNTRY')
POSTAL_CODE = os.getenv('POSTAL_CODE')
