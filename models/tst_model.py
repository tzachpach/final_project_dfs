import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_size=1
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(model_dim, output_size)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        x = self.input_proj(x)             # (batch, seq_len, model_dim)
        x = self.encoder(x)                # (batch, seq_len, model_dim)
        x = x[:, -1, :]                    # last time step
        out = self.fc(x)                   # (batch, output_size)
        return out.squeeze(-1)             # (batch,) or (batch, output_size) 