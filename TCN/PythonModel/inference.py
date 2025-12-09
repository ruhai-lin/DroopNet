"""
TinyTCN INT8 推理脚本 (PTQ 版本)

支持两种推理方式：
1. 使用 PyTorch 量化模型 (.pth)
2. 使用导出的二进制权重 (.bin) 进行手动 INT8 推理

可用于验证两种方式的推理结果一致性。
"""

import struct
import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from model import TinyTCNInt8

# ==========================================
#               配置参数
# ==========================================
BIN_PATH = "../outputs/tiny_tcn_int8.bin"
PTH_PATH = "../outputs/tiny_tcn_int8.pth"
DATA_PATH = "../../pdn_dataset_uint8.npz"

# 模型结构参数 (需与 model.py 一致)
NUM_INPUTS = 9
NUM_CHANNELS = [11, 10, 5, 4]
KERNEL_SIZE = 5
WINDOW_SIZE = 50


# ==========================================
#           Binary 文件解析器
# ==========================================
class BinaryModelReader:
    """解析 export_flat_binary 导出的 .bin 文件"""
    
    def __init__(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.data = f.read()
        self.offset = 0
        
    def read_f32(self) -> float:
        val = struct.unpack('f', self.data[self.offset:self.offset+4])[0]
        self.offset += 4
        return val
    
    def read_i32(self) -> int:
        val = struct.unpack('i', self.data[self.offset:self.offset+4])[0]
        self.offset += 4
        return val
    
    def read_f32_array(self, count: int) -> np.ndarray:
        vals = struct.unpack(f'{count}f', self.data[self.offset:self.offset+4*count])
        self.offset += 4 * count
        return np.array(vals, dtype=np.float32)
    
    def read_i32_array(self, count: int) -> np.ndarray:
        vals = struct.unpack(f'{count}i', self.data[self.offset:self.offset+4*count])
        self.offset += 4 * count
        return np.array(vals, dtype=np.int32)
    
    def read_i8_array(self, count: int) -> np.ndarray:
        vals = struct.unpack(f'{count}b', self.data[self.offset:self.offset+count])
        self.offset += count
        return np.array(vals, dtype=np.int8)


class LayerParams:
    """存储单层的量化参数和权重"""
    def __init__(self):
        self.input_scale = 0.0
        self.input_zp = 0
        self.output_scale = 0.0
        self.output_zp = 0
        self.weight_scales = None  # per-channel scales
        self.weight_zps = None     # per-channel zero points
        self.bias_int32 = None
        self.weights_int8 = None
        self.weight_shape = None


def read_layer_params(reader: BinaryModelReader, 
                      out_channels: int, 
                      weight_shape: tuple) -> LayerParams:
    """读取单层的量化参数和权重"""
    params = LayerParams()
    
    # 量化参数
    params.input_scale = reader.read_f32()
    params.input_zp = reader.read_i32()
    params.output_scale = reader.read_f32()
    params.output_zp = reader.read_i32()
    
    # 权重参数 (per-channel)
    params.weight_scales = reader.read_f32_array(out_channels)
    params.weight_zps = reader.read_i32_array(out_channels)
    params.bias_int32 = reader.read_i32_array(out_channels)
    
    # INT8 权重
    weight_count = np.prod(weight_shape)
    params.weights_int8 = reader.read_i8_array(weight_count).reshape(weight_shape)
    params.weight_shape = weight_shape
    
    return params


def load_binary_weights(filepath: str):
    """
    加载 .bin 文件中的所有权重
    
    返回结构:
    {
        'blocks': [
            {
                'conv1': LayerParams,
                'conv2': LayerParams,
                'downsample': LayerParams or None,
                'block_out_scale': float,
                'block_out_zp': int
            },
            ...
        ],
        'head': LayerParams
    }
    """
    reader = BinaryModelReader(filepath)
    model_weights = {'blocks': [], 'head': None}
    
    # Block 结构定义
    block_configs = [
        {'in_ch': NUM_INPUTS, 'out_ch': NUM_CHANNELS[0], 'dilation': 1},
        {'in_ch': NUM_CHANNELS[0], 'out_ch': NUM_CHANNELS[1], 'dilation': 2},
        {'in_ch': NUM_CHANNELS[1], 'out_ch': NUM_CHANNELS[2], 'dilation': 4},
        {'in_ch': NUM_CHANNELS[2], 'out_ch': NUM_CHANNELS[3], 'dilation': 8},
    ]
    
    for i, cfg in enumerate(block_configs):
        in_ch = cfg['in_ch']
        out_ch = cfg['out_ch']
        
        block = {}
        
        # Conv1: in_ch -> out_ch
        conv1_shape = (out_ch, in_ch, KERNEL_SIZE)
        block['conv1'] = read_layer_params(reader, out_ch, conv1_shape)
        
        # Conv2: out_ch -> out_ch
        conv2_shape = (out_ch, out_ch, KERNEL_SIZE)
        block['conv2'] = read_layer_params(reader, out_ch, conv2_shape)
        
        # Downsample: in_ch -> out_ch (1x1 conv)
        if in_ch != out_ch:
            ds_shape = (out_ch, in_ch, 1)
            block['downsample'] = read_layer_params(reader, out_ch, ds_shape)
        else:
            block['downsample'] = None
        
        # Block output scale/zp (for residual add)
        block['block_out_scale'] = reader.read_f32()
        block['block_out_zp'] = reader.read_i32()
        
        model_weights['blocks'].append(block)
    
    # Head: Linear(4, 1)
    head_shape = (1, NUM_CHANNELS[-1])
    model_weights['head'] = read_layer_params(reader, 1, head_shape)
    
    print(f"[Binary] 读取完成，共 {reader.offset} 字节")
    return model_weights


# ==========================================
#           INT8 量化运算实现
# ==========================================

def quantize_input(x_float: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """将浮点输入量化为 INT8"""
    x_q = np.round(x_float / scale + zp).astype(np.int32)
    return np.clip(x_q, -128, 127).astype(np.int8)


def dequantize_output(x_int: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """将 INT8 输出反量化为浮点"""
    return (x_int.astype(np.float32) - zp) * scale


def int8_conv1d(x_int8: np.ndarray, params: LayerParams, 
               dilation: int = 1, padding: int = 0) -> np.ndarray:
    """
    INT8 Conv1d 实现 (带 ReLU 融合)
    
    x_int8: (batch, in_channels, seq_len), int8
    weights: (out_channels, in_channels, kernel_size), int8
    
    量化卷积公式:
    y_int32 = sum((x_int8 - x_zp) * (w_int8 - w_zp)) + bias_int32
    y_float = y_int32 * (input_scale * weight_scale)
    y_int8 = quantize(relu(y_float), output_scale, output_zp)
    """
    batch, in_ch, seq_len = x_int8.shape
    out_ch, _, k = params.weights_int8.shape
    
    # 添加 padding
    if padding > 0:
        x_padded = np.pad(x_int8, ((0, 0), (0, 0), (padding, padding)), 
                         mode='constant', constant_values=params.input_zp)
    else:
        x_padded = x_int8
    
    padded_len = x_padded.shape[2]
    
    # 计算输出长度
    out_len = (padded_len - dilation * (k - 1) - 1) // 1 + 1
    
    # 转为 int32 进行计算
    x_i32 = x_padded.astype(np.int32)
    w_i32 = params.weights_int8.astype(np.int32)
    
    # 输出缓冲区
    y_int32 = np.zeros((batch, out_ch, out_len), dtype=np.int32)
    
    # 卷积计算
    for b in range(batch):
        for oc in range(out_ch):
            w_zp = params.weight_zps[oc]
            for t in range(out_len):
                acc = 0
                for ic in range(in_ch):
                    for ki in range(k):
                        idx = t + ki * dilation
                        if idx < padded_len:
                            x_val = x_i32[b, ic, idx] - params.input_zp
                            w_val = w_i32[oc, ic, ki] - w_zp
                            acc += x_val * w_val
                
                # 加上量化后的 bias
                acc += params.bias_int32[oc]
                y_int32[b, oc, t] = acc
    
    # 反量化 + ReLU + 重新量化
    y_float = np.zeros_like(y_int32, dtype=np.float32)
    for oc in range(out_ch):
        scale_factor = params.input_scale * params.weight_scales[oc]
        y_float[:, oc, :] = y_int32[:, oc, :] * scale_factor
    
    # ReLU
    y_float = np.maximum(y_float, 0)
    
    # 重新量化到输出 scale
    y_out = np.round(y_float / params.output_scale + params.output_zp).astype(np.int32)
    y_out = np.clip(y_out, -128, 127).astype(np.int8)
    
    return y_out


def int8_conv1d_no_relu(x_int8: np.ndarray, params: LayerParams, 
                        dilation: int = 1, padding: int = 0) -> np.ndarray:
    """INT8 Conv1d 实现 (不带 ReLU)"""
    batch, in_ch, seq_len = x_int8.shape
    out_ch, _, k = params.weights_int8.shape
    
    if padding > 0:
        x_padded = np.pad(x_int8, ((0, 0), (0, 0), (padding, padding)), 
                         mode='constant', constant_values=params.input_zp)
    else:
        x_padded = x_int8
    
    padded_len = x_padded.shape[2]
    out_len = (padded_len - dilation * (k - 1) - 1) // 1 + 1
    
    x_i32 = x_padded.astype(np.int32)
    w_i32 = params.weights_int8.astype(np.int32)
    
    y_int32 = np.zeros((batch, out_ch, out_len), dtype=np.int32)
    
    for b in range(batch):
        for oc in range(out_ch):
            w_zp = params.weight_zps[oc]
            for t in range(out_len):
                acc = 0
                for ic in range(in_ch):
                    for ki in range(k):
                        idx = t + ki * dilation
                        if idx < padded_len:
                            x_val = x_i32[b, ic, idx] - params.input_zp
                            w_val = w_i32[oc, ic, ki] - w_zp
                            acc += x_val * w_val
                acc += params.bias_int32[oc]
                y_int32[b, oc, t] = acc
    
    y_float = np.zeros_like(y_int32, dtype=np.float32)
    for oc in range(out_ch):
        scale_factor = params.input_scale * params.weight_scales[oc]
        y_float[:, oc, :] = y_int32[:, oc, :] * scale_factor
    
    y_out = np.round(y_float / params.output_scale + params.output_zp).astype(np.int32)
    y_out = np.clip(y_out, -128, 127).astype(np.int8)
    
    return y_out


def int8_linear(x_int8: np.ndarray, params: LayerParams) -> np.ndarray:
    """
    INT8 Linear 实现
    x_int8: (batch, in_features), int8
    """
    batch, in_features = x_int8.shape
    out_features = params.weights_int8.shape[0]
    
    x_i32 = x_int8.astype(np.int32)
    w_i32 = params.weights_int8.astype(np.int32)
    
    y_int32 = np.zeros((batch, out_features), dtype=np.int32)
    
    for b in range(batch):
        for oc in range(out_features):
            w_zp = params.weight_zps[oc]
            acc = 0
            for ic in range(in_features):
                x_val = x_i32[b, ic] - params.input_zp
                w_val = w_i32[oc, ic] - w_zp
                acc += x_val * w_val
            acc += params.bias_int32[oc]
            y_int32[b, oc] = acc
    
    # 反量化
    y_float = np.zeros_like(y_int32, dtype=np.float32)
    for oc in range(out_features):
        scale_factor = params.input_scale * params.weight_scales[oc]
        y_float[:, oc] = y_int32[:, oc] * scale_factor
    
    return y_float


def int8_requantize(x_int8: np.ndarray, 
                    old_scale: float, old_zp: int,
                    new_scale: float, new_zp: int) -> np.ndarray:
    """重新量化到新的 scale/zp"""
    x_float = (x_int8.astype(np.float32) - old_zp) * old_scale
    x_new = np.round(x_float / new_scale + new_zp).astype(np.int32)
    return np.clip(x_new, -128, 127).astype(np.int8)


def int8_add_relu(a_int8: np.ndarray, b_int8: np.ndarray,
                  a_scale: float, a_zp: int,
                  b_scale: float, b_zp: int,
                  out_scale: float, out_zp: int) -> np.ndarray:
    """INT8 残差相加 + ReLU"""
    # 反量化
    a_float = (a_int8.astype(np.float32) - a_zp) * a_scale
    b_float = (b_int8.astype(np.float32) - b_zp) * b_scale
    
    # 相加 + ReLU
    y_float = np.maximum(a_float + b_float, 0)
    
    # 重新量化
    y_out = np.round(y_float / out_scale + out_zp).astype(np.int32)
    return np.clip(y_out, -128, 127).astype(np.int8)


# ==========================================
#           Binary 权重推理器
# ==========================================

class BinaryInference:
    """使用 .bin 文件进行 INT8 推理"""
    
    def __init__(self, weights_dict):
        self.weights = weights_dict
        self.dilations = [1, 2, 4, 8]
    
    def forward_block(self, x_int8: np.ndarray, block_idx: int,
                      current_scale: float, current_zp: int):
        """前向传播单个 Block"""
        block = self.weights['blocks'][block_idx]
        dilation = self.dilations[block_idx]
        padding = (KERNEL_SIZE - 1) * dilation
        
        conv1_params = block['conv1']
        conv2_params = block['conv2']
        
        # Conv1 + ReLU (带 chomp - 通过只取前 seq_len 个输出实现)
        out = int8_conv1d(x_int8, conv1_params, dilation=dilation, padding=padding)
        seq_len = x_int8.shape[2]
        out = out[:, :, :seq_len]  # Chomp
        
        # Conv2 + ReLU (带 chomp)
        out = int8_conv1d(out, conv2_params, dilation=dilation, padding=padding)
        out = out[:, :, :seq_len]  # Chomp
        
        # 残差连接
        if block['downsample'] is not None:
            ds_params = block['downsample']
            residual = int8_conv1d_no_relu(x_int8, ds_params, dilation=1, padding=0)
        else:
            residual = x_int8
        
        # Add + ReLU
        out_int8 = int8_add_relu(
            residual, out,
            block['downsample'].output_scale if block['downsample'] else current_scale,
            block['downsample'].output_zp if block['downsample'] else current_zp,
            conv2_params.output_scale, conv2_params.output_zp,
            block['block_out_scale'], block['block_out_zp']
        )
        
        return out_int8, block['block_out_scale'], block['block_out_zp']
    
    def forward(self, x_float: np.ndarray) -> np.ndarray:
        """
        完整前向传播
        x_float: (batch, seq_len, channels), float32 [0, 1]
        """
        # 转置为 (batch, channels, seq_len)
        x = x_float.transpose(0, 2, 1)
        
        # 获取初始量化参数 (从第一个 conv1 的输入参数获取)
        first_conv1 = self.weights['blocks'][0]['conv1']
        input_scale = first_conv1.input_scale
        input_zp = first_conv1.input_zp
        
        # 量化输入
        x_int8 = quantize_input(x, input_scale, input_zp)
        
        current_scale = input_scale
        current_zp = input_zp
        
        # 通过各个 Block
        for i in range(4):
            x_int8, current_scale, current_zp = self.forward_block(
                x_int8, i, current_scale, current_zp
            )
        
        # 取最后一个时间步
        x_last = x_int8[:, :, -1]  # (batch, channels)
        
        # Head (Linear)
        output = int8_linear(x_last, self.weights['head'])
        
        return output


# ==========================================
#           PyTorch .pth 推理
# ==========================================

def load_pth_model(filepath: str) -> nn.Module:
    """加载 PyTorch PTQ 量化模型"""
    model = TinyTCNInt8()
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(model, inplace=True)
    model.eval()
    tq.convert(model, inplace=True)
    
    state_dict = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def pth_inference(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """使用 PyTorch 模型推理"""
    x_tensor = torch.from_numpy(x).float()
    with torch.no_grad():
        output = model(x_tensor)
    return output.numpy()


# ==========================================
#           主函数
# ==========================================

def main():
    print("=" * 60)
    print("TinyTCN INT8 推理对比测试 (PTQ 版本)")
    print("=" * 60)
    
    # 1. 加载测试数据
    print("\n[1] 加载测试数据...")
    data = np.load(DATA_PATH)
    X = data['X']  # uint8
    y = data['y']
    
    # 选择一部分样本进行测试
    N_TEST = 100
    X_test = X[:N_TEST].astype(np.float32) / 255.0  # 归一化到 [0, 1]
    y_test = y[:N_TEST]
    
    print(f"   测试样本数: {N_TEST}")
    print(f"   输入形状: {X_test.shape}")  # (N, 50, 9)
    
    # 2. 加载 PyTorch 量化模型 (.pth)
    print("\n[2] 加载 PyTorch 量化模型 (.pth)...")
    try:
        pth_model = load_pth_model(PTH_PATH)
        print("   加载成功!")
    except Exception as e:
        print(f"   加载失败: {e}")
        pth_model = None
    
    # 3. 加载二进制权重 (.bin)
    print("\n[3] 加载二进制权重 (.bin)...")
    try:
        bin_weights = load_binary_weights(BIN_PATH)
        bin_inference = BinaryInference(bin_weights)
        print("   加载成功!")
    except Exception as e:
        print(f"   加载失败: {e}")
        bin_inference = None
    
    # 4. 进行推理对比
    print("\n[4] 推理对比...")
    print("-" * 60)
    
    if pth_model is not None:
        print("\n>>> PyTorch (.pth) 推理:")
        pth_outputs = pth_inference(pth_model, X_test)
        pth_probs = 1 / (1 + np.exp(-pth_outputs))  # Sigmoid
        pth_preds = (pth_probs > 0.5).astype(np.float32).flatten()
        
        pth_acc = (pth_preds == y_test).mean()
        print(f"    准确率: {pth_acc:.4f}")
        print(f"    输出范围: [{pth_outputs.min():.4f}, {pth_outputs.max():.4f}]")
        print(f"    输出均值: {pth_outputs.mean():.4f}")
    
    if bin_inference is not None:
        print("\n>>> Binary (.bin) INT8 推理:")
        bin_outputs = bin_inference.forward(X_test)
        bin_probs = 1 / (1 + np.exp(-bin_outputs))  # Sigmoid
        bin_preds = (bin_probs > 0.5).astype(np.float32).flatten()
        
        bin_acc = (bin_preds == y_test).mean()
        print(f"    准确率: {bin_acc:.4f}")
        print(f"    输出范围: [{bin_outputs.min():.4f}, {bin_outputs.max():.4f}]")
        print(f"    输出均值: {bin_outputs.mean():.4f}")
    
    # 5. 对比两种方式的差异
    if pth_model is not None and bin_inference is not None:
        print("\n[5] 两种推理方式对比:")
        print("-" * 60)
        
        # 输出差异
        output_diff = np.abs(pth_outputs.flatten() - bin_outputs.flatten())
        print(f"    输出绝对差异:")
        print(f"      - 最大值: {output_diff.max():.6f}")
        print(f"      - 平均值: {output_diff.mean():.6f}")
        print(f"      - 中位数: {np.median(output_diff):.6f}")
        
        # 预测一致性
        pred_match = (pth_preds == bin_preds).mean()
        print(f"\n    预测一致性: {pred_match:.4f} ({int(pred_match * N_TEST)}/{N_TEST})")
        
        # 准确率差异
        acc_diff = abs(pth_acc - bin_acc)
        print(f"    准确率差异: {acc_diff:.4f}")
        
        # 相关系数
        correlation = np.corrcoef(pth_outputs.flatten(), bin_outputs.flatten())[0, 1]
        print(f"    输出相关系数: {correlation:.6f}")
        
        # 判断差异是否可接受
        print("\n" + "=" * 60)
        if output_diff.max() < 0.5 and pred_match > 0.95:
            print("✓ 结论: 两种推理方式结果高度一致，.bin 文件可以正常使用!")
        elif output_diff.max() < 1.0 and pred_match > 0.9:
            print("△ 结论: 两种推理方式存在轻微差异，可能是量化精度损失造成的")
        else:
            print("✗ 结论: 两种推理方式存在较大差异，请检查权重导出/加载逻辑")
    
    # 6. 单样本详细对比
    print("\n[6] 单样本详细对比 (前5个样本):")
    print("-" * 60)
    print(f"{'Sample':<8} {'Label':<8} {'PTH Output':<15} {'BIN Output':<15} {'Diff':<10}")
    print("-" * 60)
    
    for i in range(min(5, N_TEST)):
        pth_out = pth_outputs[i, 0] if pth_model else float('nan')
        bin_out = bin_outputs[i, 0] if bin_inference else float('nan')
        diff = abs(pth_out - bin_out) if (pth_model and bin_inference) else float('nan')
        print(f"{i:<8} {y_test[i]:<8.0f} {pth_out:<15.6f} {bin_out:<15.6f} {diff:<10.6f}")
    
    print("\n" + "=" * 60)
    print("推理对比完成!")


if __name__ == "__main__":
    main()

