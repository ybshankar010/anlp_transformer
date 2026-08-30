
# from .utils import Logger
import logging
from typing import Any
import numpy as np
import os


from collections import Counter
from src.constants import PLAIN_TEXT_PATH,CIPHER_TEXT_PATH
from src.utils import get_lines_from_file_path
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

HUGGINGFACE_EMBEDDING_EABLED = (
    os.getenv("HUGGINGFACE_EMBEDDING_EABLED", "false").lower()=="true"
)

#EDA 
class EDA:

    def __init__(self):
        self._files = [CIPHER_TEXT_PATH,PLAIN_TEXT_PATH]

    def get_line_counts(self, content):
        logger.debug("#lines %s ", len(content))

        empty_line_count = sum(1 for line in content if ((len(line.strip()) == 0)))
        logger.debug("Empty line count = %s",empty_line_count)

    def get_whitespace_counts(self,content):
        whitespace_chars = sorted({ch for line in content for ch in line if ch.isspace()})
        logger.debug("Whitespace characters counts %s",len(whitespace_chars))
        logger.debug("Whitespace characters %s",whitespace_chars)

    def get_unique_character_counts(self,content):
        character_map = Counter(
            ch
            for line in content
            for ch in line
            if not ch.isspace()
        )
    
        logger.debug("Unique characters counts %s",len(character_map))
        logger.debug("Unique characters %s",character_map)

    def get_line_length_stats(self, content):
        line_lengths = [len(line) for line in content]
        min_len,max_len, avg_len = np.min(line_lengths), np.max(line_lengths), np.average(line_lengths)
        p50_len, p75_len, p90_len = np.percentile(line_lengths,50), np.percentile(line_lengths,75),np.percentile(line_lengths,90)

        logger.debug("Min line length %s ", min_len)
        logger.debug("Max line length %s ", max_len)
        logger.debug("Average line length %s ", avg_len)
        logger.debug("p50 line length %s ", p50_len)
        logger.debug("p75 line length %s ", p75_len)
        logger.debug("p90 line length %s ", p90_len)

    def get_dataset_stats(self):

        for file in self._files:
            logger.info("="*50)
            logger.info("%s file stats", file)
            lines = get_lines_from_file_path(file_path=file)
            self.get_line_counts(content=lines)
            self.get_whitespace_counts(content=lines)
            self.get_unique_character_counts(content=lines)
            self.get_line_length_stats(content=lines)



#Dataset classes
from torch.utils.data import Dataset,random_split,DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from .utils import ExperimentConfig,encode_text, EXPERIMENTS
from .tokenizer import BPETokenizer
import torch

class CipherPlainDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.cipher_data = get_lines_from_file_path(CIPHER_TEXT_PATH)
        self.plain_data = get_lines_from_file_path(PLAIN_TEXT_PATH)

        self.cipher_data = [line.rstrip("\n") for line in self.cipher_data]
        self.plain_data = [line.rstrip("\n") for line in self.plain_data]
        
        if len(self.cipher_data) != len(self.plain_data):
            logger.error("Invalid dataset")

    def __getitem__(self, index) -> Any:
        plain = self.plain_data[index]
        cipher = self.cipher_data[index]

        return {
            "cipher_text" : cipher,
            "plain_text" : plain
            }
        
    def __len__(self)-> int:
        return len(self.cipher_data)


class CipherPlainDatasetCollator: 

    def __init__(self,cipher_tokenizer: BPETokenizer,plain_text_tokenizer: BPETokenizer,src_pad_id = 0, max_target_len = 256) -> None:

        self.cipher_tokenizer = cipher_tokenizer
        self.plain_text_tokenizer = plain_text_tokenizer
        self.bos_id = self.plain_text_tokenizer.bos_token_id
        self.eos_id = self.plain_text_tokenizer.eos_token_id
        self.pad_id = self.plain_text_tokenizer.pad_token_id
            

        self.src_pad_id = cipher_tokenizer.pad_token_id
        self.max_target_len = max_target_len


    def __call__(self, batch) -> Any:
        cipher_ids = [
            torch.tensor(self.cipher_tokenizer.encode(item["cipher_text"]),dtype=torch.long) for item in batch
        ]
        plain_text = [item["plain_text"] for item in batch] 
        padded_sequence = pad_sequence(
            cipher_ids,
            batch_first=True,
            padding_value=self.src_pad_id
        )
        cipher_padding_mask = (padded_sequence == self.src_pad_id)

        plain_text_ids = [self.plain_text_tokenizer.encode(text) for text in plain_text]

        plain_text_input_ids = [torch.tensor(([self.bos_id] + ids[: self.max_target_len - 1]),dtype=torch.long) for ids in plain_text_ids]
        plain_text_padded_input_ids = pad_sequence(plain_text_input_ids,batch_first = True,padding_value = self.pad_id)
        plain_text_target_ids = [torch.tensor((ids[: self.max_target_len - 1] + [self.eos_id]),dtype=torch.long) for ids in plain_text_ids]
        plain_text_padded_target_ids = pad_sequence(plain_text_target_ids,batch_first = True,padding_value = self.pad_id)

        plain_text_padding_mask = plain_text_padded_input_ids == self.pad_id

        return {
            "cipher_text" : padded_sequence,
            "cipher_padding_mask" : cipher_padding_mask,
            "plain_text" : plain_text,
            "plain_text_input_ids" : plain_text_padded_input_ids,
            "plain_text_target_ids" : plain_text_padded_target_ids,
            "plain_text_padding_mask" : plain_text_padding_mask
        }


def create_datasplits(dataset: CipherPlainDataset, config: ExperimentConfig):
    total_length = len(dataset)

    train_size = int(config.train_split_ratio * total_length)
    validation_size = int(config.validation_split_ratio * total_length)
    test_size = total_length - train_size - validation_size

    generator = torch.Generator().manual_seed(config.seed)

    return random_split(
        dataset,
        [train_size,validation_size,test_size],
        generator=generator
    )



def test():
    dataset = CipherPlainDataset()
    train_dataset, _, _ = create_datasplits(dataset,EXPERIMENTS[0])

    cipher_dataset = [item["cipher_text"] for item in train_dataset]
    plain_dataset = [item["plain_text"] for item in train_dataset]

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    cipher_tokenizer = BPETokenizer(vocab_size=100)
    plain_tokenizer = BPETokenizer(vocab_size=100)

    cipher_tokenizer.train(cipher_dataset[:100])
    plain_tokenizer.train(plain_dataset[:100])

    collator = CipherPlainDatasetCollator(
        cipher_tokenizer=cipher_tokenizer,
        plain_text_tokenizer=plain_tokenizer,
        max_target_len=EXPERIMENTS[0].max_target_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size= EXPERIMENTS[0].batch_size,
        shuffle=True,
        collate_fn=collator
    )

    batch = next(iter(train_loader))

    logger.debug(batch["cipher_text"].shape)
    logger.debug(batch["cipher_padding_mask"].shape)
    logger.debug(batch["plain_text"][0][:100])
    logger.debug(batch["plain_text_input_ids"].shape)
    logger.debug(batch["plain_text_target_ids"].shape)
    logger.debug(batch["plain_text_padding_mask"].shape)



