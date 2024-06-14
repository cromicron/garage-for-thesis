import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=256,
        emb_size=10,
    ):
        super(GRUEncoder, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._emb_size = emb_size
        self._gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self._fc = nn.Linear(hidden_size, emb_size)

    def forward(self, x, hidden=None):
        x, hidden = self._gru(x, hidden)
        embedding = self._fc(x)
        return embedding, hidden
