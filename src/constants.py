
from bidict import bidict

PLAIN_TEXT_PATH = "./Dataset_A1/brown_plain.txt"
CIPHER_TEXT_PATH = "./Dataset_A1/brown_cipher.txt"

SRC_TOKEN_TO_ID_MAP = bidict({
        "<PAD>" : 0,
        "0" : 1,
        "1" : 2
    })