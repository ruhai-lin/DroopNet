import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.nn.quantized import FloatFunctional


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()


class QuantTemporalBlock(nn.Module):
    """
    Minimal two-conv TCN block with quantization-friendly ops.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
            if in_channels != out_channels
            else None
        )
        self.skip_add = FloatFunctional()
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)

        out = self.skip_add.add(residual, out)
        return self.out_relu(out)

    def fuse_model(self) -> None:
        tq.fuse_modules(self, [["conv1", "relu1"], ["conv2", "relu2"]], inplace=True)
        # Downsample has no fused activation; skip_add is quantization-friendly FloatFunctional.


class TinyTCNInt8(nn.Module):
    """
    Compact TCN (4 levels) tuned for ~4 KB weights and ~2 KB activations
    at seq_len≈50, batch=1 when quantized to int8.
    """

    def __init__(
        self,
        num_inputs: int = 9,
        num_channels=None,
        kernel_size: int = 5,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [11, 10, 5, 4]

        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        levels = []
        for i, ch in enumerate(num_channels):
            dilation = 2 ** i  # 1, 2, 4, 8
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            levels.append(QuantTemporalBlock(in_ch, ch, kernel_size, dilation))

        self.network = nn.Sequential(*levels)
        self.head = nn.Linear(num_channels[-1], 1, bias=True)

        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives as uint8 ADC codes of shape (B, T, C). Normalize to float.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.shape[2] == self.num_inputs:
            x = x.permute(0, 2, 1)  # (B, C, T)

        x = self.quant(x)
        y = self.network(x)
        y = y[:, :, -1]  # last timestep
        y = self.head(y)
        return self.dequant(y)

    def fuse_model(self) -> None:
        for block in self.network:
            if hasattr(block, "fuse_model"):
                block.fuse_model()

    def footprint_bytes(self, seq_len: int = 50, batch_size: int = 1) -> dict:
        """
        Footprint estimate assuming int8 weights/activations.
        Scale/zero-point tensors are ignored (small relative to 8 KB budget).
        """
        weight_count = sum(p.numel() for p in self.parameters())  # int8 params
        weight_bytes = weight_count  # 1 byte each once quantized

        act_counts = [self.num_inputs * seq_len]
        act_counts.extend(ch * seq_len for ch in self.num_channels)
        activation_bytes = batch_size * sum(act_counts)  # int8 activations

        return {
            "weight_bytes": weight_bytes,
            "activation_bytes": activation_bytes,
            "total_bytes": weight_bytes + activation_bytes,
        }

