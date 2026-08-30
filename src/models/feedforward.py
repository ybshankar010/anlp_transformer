import torch
import torch.nn as nn
import logging


logger = logging.getLogger(__name__)

class PositionwiseFeedforward(nn.Module):
    def __init__(self,d_model, ffn_dim, dropout = 0.1 ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim,d_model)


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x



def test_ffn():
    ffn = PositionwiseFeedforward(d_model=256, ffn_dim=1024)

    x = torch.randn(16,100,256)
    y = ffn(x)

    logger.debug("=============TEST FFN=============")
    logger.debug("FFN output shape %s",y.shape)

    return