import math
from collections.abc import Sequence

import torch
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        channels: Sequence[int] = (256, 256, 256),
        kernel_sizes: Sequence[int] = (5, 5, 5),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(channels) == len(kernel_sizes)

        layers: list[nn.Module] = []
        in_channels = num_features
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.out_features = channels[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.permute(1, 2, 0)
        x = self.cnn(x)
        return x.permute(2, 0, 1)


class RNNEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.out_features = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Linear(self.out_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(inputs)
        x = self.projection(x)
        return self.layer_norm(x)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.out_features = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Linear(self.out_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        with torch.backends.cudnn.flags(enabled=False):
            x, _ = self.lstm(inputs.contiguous())
        x = self.projection(x)
        return self.layer_norm(x)


class GRUEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.out_features = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Linear(self.out_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        with torch.backends.cudnn.flags(enabled=False):
            x, _ = self.gru(inputs.contiguous())
        x = self.projection(x)
        return self.layer_norm(x)


class CNNRNNEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int] = (256, 256),
        cnn_kernel_sizes: Sequence[int] = (5, 5),
        rnn_hidden_size: int = 384,
        rnn_num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(num_features, cnn_channels, cnn_kernel_sizes, dropout)
        self.rnn = RNNEncoder(
            self.cnn.out_features, rnn_hidden_size, rnn_num_layers, dropout, bidirectional
        )
        self.projection = nn.Linear(self.cnn.out_features, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(inputs)
        x = self.rnn(x)
        return self.projection(x)


class CNNLSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int] = (256, 256),
        cnn_kernel_sizes: Sequence[int] = (5, 5),
        lstm_hidden_size: int = 384,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(num_features, cnn_channels, cnn_kernel_sizes, dropout)
        self.lstm = LSTMEncoder(
            self.cnn.out_features, lstm_hidden_size, lstm_num_layers, dropout, bidirectional
        )
        self.projection = nn.Linear(self.cnn.out_features, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(inputs)
        x = self.lstm(x)
        return self.projection(x)


class CNNGRUEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int] = (256, 256),
        cnn_kernel_sizes: Sequence[int] = (5, 5),
        gru_hidden_size: int = 384,
        gru_num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(num_features, cnn_channels, cnn_kernel_sizes, dropout)
        self.gru = GRUEncoder(
            self.cnn.out_features, gru_hidden_size, gru_num_layers, dropout, bidirectional
        )
        self.projection = nn.Linear(self.cnn.out_features, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(inputs)
        x = self.gru(x)
        return self.projection(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(num_features, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(inputs)
        x = self.transformer(x)
        return self.layer_norm(x)


class CNNTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_channels: Sequence[int] = (256, 256),
        cnn_kernel_sizes: Sequence[int] = (5, 5),
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(num_features, cnn_channels, cnn_kernel_sizes, dropout)
        self.input_projection = nn.Linear(self.cnn.out_features, num_features)
        self.transformer = TransformerEncoder(
            num_features=num_features,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(inputs)
        x = self.input_projection(x)
        return self.transformer(x)
