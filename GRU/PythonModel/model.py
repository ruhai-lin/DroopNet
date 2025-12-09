"""
TinyGRU - INT8 量化友好的 GRU 模型用于 PDN Droop 预测
目标: 权重 + 激活 < 6KB (ASIC 8KB SRAM)

GRU 参数计算:
- weight_ih: (3*H, I) -> 3*H*I
- weight_hh: (3*H, H) -> 3*H*H
- bias_ih + bias_hh: 6*H
- 总参数: 3*H*(I + H + 2)

默认配置 (H=36, I=9):
- GRU权重: 3*36*(9+36+2) = 5076 bytes
- Head权重: 36+1 = 37 bytes
- 激活(seq=50): 50*9 + 36 = 486 bytes
- 总计: ~5.6KB
"""

import torch
import torch.nn as nn
import torch.ao.quantization as tq


class TinyGRUInt8(nn.Module):
    """
    紧凑型 GRU 模型，专为 INT8 量化和 ASIC 部署设计。
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
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
        )
        
        # 分类头
        self.head = nn.Linear(hidden_size, 1, bias=True)
        
        # 量化 Stubs (用于校准)
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
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
        前向传播
        
        Args:
            x: 输入张量，形状 (B, T, C) 或 uint8 ADC 码
            
        Returns:
            logits: 形状 (B, 1) 的预测 logits
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
        """估算 INT8 量化后的内存占用"""
        # GRU 权重: weight_ih (3*H, I) + weight_hh (3*H, H) + bias (6*H)
        gru_weights = 3 * self.hidden_size * (self.input_size + self.hidden_size + 2)
        head_weights = self.hidden_size + 1
        total_weight_bytes = gru_weights + head_weights
        
        # 激活值: 输入 + 隐藏状态
        input_act = seq_len * self.input_size * batch_size
        hidden_act = self.hidden_size * batch_size
        total_activation_bytes = input_act + hidden_act
        
        return {
            "weight_bytes": total_weight_bytes,
            "activation_bytes": total_activation_bytes,
            "total_bytes": total_weight_bytes + total_activation_bytes,
        }
    
    def get_export_info(self) -> dict:
        """获取导出信息"""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
