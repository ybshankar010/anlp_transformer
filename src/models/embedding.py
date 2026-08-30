import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self,vocab_size,d_model,pad_id):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id
        )

        self.scale = math.sqrt(d_model)


    def forward(self,token_ids):
        return self.embedding(token_ids) * self.scale