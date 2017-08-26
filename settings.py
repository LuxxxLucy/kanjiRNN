"""App settings, typically drawn from env vars."""

from os import environ
from os.path import join, dirname
from dotenv import Dotenv

_DOTENV_PATH = join(dirname(__file__), '.env')
dotenv = Dotenv(_DOTENV_PATH)
environ.update(dotenv)

DATA_STORE_PATH = str(environ.get('DATA_STORE_PATH', 'data_store'))
if not DATA_STORE_PATH.startswith('/'):
    DATA_STORE_PATH = dirname(__file__) + '/' + DATA_STORE_PATH

MODEL_STORE_PATH = str(environ.get('MODEL_STORE_PATH', 'model_store'))
if not MODEL_STORE_PATH.startswith('/'):
    MODEL_STORE_PATH = dirname(__file__) + '/' + MODEL_STORE_PATH
