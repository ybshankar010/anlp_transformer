from bidict import bidict
from collections import Counter
import logging
import json

logger = logging.getLogger(__name__)

class BPETokenizer:
    def __init__(self, vocab_size = 1000, special_tokens = None) -> None:
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        if special_tokens is None:
            special_tokens = [self.pad_token,self.unk_token,self.bos_token,self.eos_token]

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.token_to_id = bidict({})
        self.merges = []

        for token in self.special_tokens:
            self._add_token(token=token)


    def _add_token(self,token):
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index

    def pre_tokenize(self, text) : 
        return list(text)
    
    def encode(self, text,add_special_tokens=False):
        tokens = self.pre_tokenize(text)

        for best_pair, token in self.merges:
            tokens = self.merge_pair_in_tokens(tokens, best_pair,token)


        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]
        return token_ids

    def decode(self,ids):
        tokens = [self.token_to_id.inverse[idx] for idx in ids]
        tokens = [token for token in tokens if token not in self.special_tokens]
        return "".join(tokens)

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "token_to_id": dict(self.token_to_id),
            "merges": [
                {
                    "pair": list(pair),
                    "token": token,
                }
                for pair, token in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, data):
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
        )
        tokenizer.token_to_id = bidict(data["token_to_id"])
        tokenizer.merges = [
            (tuple(merge["pair"]), merge["token"])
            for merge in data["merges"]
        ]
        return tokenizer

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def train(self, texts):
        #Simple tokenization
        tokenized_texts = [self.pre_tokenize(text) for text in texts]

        ## Adding to vocab
        for tokens in tokenized_texts:
            for token in tokens: 
                self._add_token(token=token)

        while (len(self.token_to_id) < self.vocab_size):
            pair_counts = self.get_pair_counts(tokenized_texts)

            if not pair_counts:
                break

            best_pair, best_pair_count = pair_counts.most_common(1)[0]
            new_token = "".join(best_pair)
            logger.debug("Most frequent Pair %s with count %s creating a new token %s",best_pair,best_pair_count,new_token)

            if new_token in self.token_to_id:
                break

            self._add_token(new_token)
            self.merges.append((best_pair,new_token))

            tokenized_texts = [ self.merge_pair_in_tokens(token_text,best_pair,new_token) for token_text in tokenized_texts]

        return

        
    def get_pair_counts(self, tokenized_texts):
        pairs = []
        for single_line in tokenized_texts:
            line_length = len(single_line)
            for index in range(0,line_length-1):
                curr_word, next_word = single_line[index],single_line[index+1]
                pairs.append((curr_word,next_word))

        return Counter(pairs)

    def merge_pair_in_tokens(self, tokens, pair, new_token):
        new_tokens = []
        index = 0

        while(index < len(tokens)):

            if ((index < len(tokens)-1) and (tokens[index] == pair[0]) and
                 (tokens[index+1] == pair[1])):
                new_tokens.append(new_token)
                index = index + 2
            else :
                new_tokens.append(tokens[index])
                index = index+1

        return new_tokens

    def get_pad_id(self):
        return self.token_to_id[self.pad_token]

    def get_unk_id(self):
        return self.token_to_id[self.unk_token]

    def get_bos_id(self):
        return self.token_to_id[self.bos_token]

    def get_eos_id(self):
        return self.token_to_id[self.eos_token]

    def get_vocab_size(self):
        return len(self.token_to_id)

    @property
    def pad_token_id(self):
        return self.get_pad_id()

    @property
    def bos_token_id(self):
        return self.get_bos_id()

    @property
    def eos_token_id(self):
        return self.get_eos_id()

    @property
    def unk_token_id(self):
        return self.get_unk_id()

    @property
    def cls_token_id(self):
        return self.get_bos_id()

    @property
    def sep_token_id(self):
        return self.get_eos_id()
