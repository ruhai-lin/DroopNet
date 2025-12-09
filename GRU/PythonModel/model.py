"""
TinyGRU - INT8-friendly GRU model for PDN droop prediction
Goal: weights + activations < 6KB (ASIC 8KB SRAM)

GRU parameter breakdown:
- weight_ih: (3*H, I) -> 3*H*I
- weight_hh: (3*H, H) -> 3*H*H
- bias_ih + bias_hh: 6*H
- Total params: 3*H*(I + H + 2)

Default config (H=36, I=9):
- GRU weights: 3*36*(9+36+2) = 5076 bytes
- Head weights: 36+1 = 37 bytes
- Activations (seq=50): 50*9 + 36 = 486 bytes
- Total: ~5.6KB
"""

import torch
import torch.nn as nn
import torch.ao.quantization as tq


class TinyGRUInt8(nn.Module):
    """
    Compact GRU model designed for INT8 quantization and ASIC deployment.
    """
    
    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 36,
        num_layers: int = 1,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
        )
        
        # Classification head
        self.head = nn.Linear(hidden_size, 1, bias=True)
        
        # Quantization stubs (for calibration)
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: input tensor, shape (B, T, C) or uint8 ADC codes
            
        Returns:
            logits: predicted logits of shape (B, 1)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = self.quant(x)
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        logits = self.head(last_output)
        return self.dequant(logits)
    
    def footprint_bytes(self, seq_len: int = 50, batch_size: int = 1) -> dict:
        """Estimate memory footprint after INT8 quantization"""
        # GRU weights: weight_ih (3*H, I) + weight_hh (3*H, H) + bias (6*H)
        gru_weights = 3 * self.hidden_size * (self.input_size + self.hidden_size + 2)
        head_weights = self.hidden_size + 1
        total_weight_bytes = gru_weights + head_weights
        
        # Activations: input + hidden state
        input_act = seq_len * self.input_size * batch_size
        hidden_act = self.hidden_size * batch_size
        total_activation_bytes = input_act + hidden_act
        
        return {
            "weight_bytes": total_weight_bytes,
            "activation_bytes": total_activation_bytes,
            "total_bytes": total_weight_bytes + total_activation_bytes,
        }
    
    def get_export_info(self) -> dict:
        """Get export info"""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
