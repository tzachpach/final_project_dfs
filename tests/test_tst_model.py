import torch
from models.tst_model import TimeSeriesTransformer

def test_tst_forward_pass():
    batch_size = 4
    seq_len = 5
    input_size = 8
    model_dim = 16
    num_heads = 2
    num_layers = 1
    output_size = 1
    x = torch.randn(batch_size, seq_len, input_size)
    model = TimeSeriesTransformer(
        input_size=input_size,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        output_size=output_size,
    )
    out = model(x)
    assert out.shape[0] == batch_size
    if output_size == 1:
        assert out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1)
    else:
        assert out.shape[1] == output_size 