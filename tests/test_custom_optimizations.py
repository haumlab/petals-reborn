import pytest
import torch
import torch.nn as nn
from petals.utils.convert_block import CPUAndMPSLinear8bit, CPUAndMPSLinear4bit, quantize_module, QuantType

def test_linear8bit():
    linear = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    y_orig = linear(x)
    quantized = CPUAndMPSLinear8bit.from_linear(linear)
    y_quant = quantized(x)
    diff = (y_orig - y_quant).abs().mean().item()
    assert diff < 0.1
    assert quantized.weight.dtype == torch.int8

def test_linear4bit():
    for in_features in [32, 31]:
        linear = nn.Linear(in_features, 16)
        x = torch.randn(4, in_features)
        y_orig = linear(x)
        quantized = CPUAndMPSLinear4bit.from_linear(linear)
        y_quant = quantized(x)
        diff = (y_orig - y_quant).abs().mean().item()
        assert diff < 0.3
        assert quantized.weight_packed.dtype == torch.uint8

def test_quantize_module_fallback():
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 8)
            self.lm_head = nn.Linear(8, 4)
    model = SimpleModule()
    quantized_model = quantize_module(model, quant_type=QuantType.INT8)
    assert isinstance(quantized_model.fc1, CPUAndMPSLinear8bit)
    assert isinstance(quantized_model.fc2, CPUAndMPSLinear8bit)
    assert isinstance(quantized_model.lm_head, nn.Linear)
