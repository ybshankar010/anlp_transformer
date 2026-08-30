
from bidict import bidict

PLAIN_TEXT_PATH = "./Dataset_A1/brown_plain.txt"
CIPHER_TEXT_PATH = "./Dataset_A1/brown_cipher.txt"

SRC_TOKEN_TO_ID_MAP = bidict({
        "<PAD>" : 0,
        "0" : 1,
        "1" : 2
    })

BIT_ZERO_ID = 0
BIT_ONE_ID = 1
BIT_PAD_ID = 2
BIT_VOCAB_SIZE = 3

BYTE_PAD_ID = 256
BYTE_BOS_ID = 257
BYTE_EOS_ID = 258
BYTE_VOCAB_SIZE = 259
