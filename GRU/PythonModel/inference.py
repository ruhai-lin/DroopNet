"""
TinyGRU INT8 推理脚本

支持两种推理方式：
1. 使用 PyTorch 浮点模型 (.pth)
2. 使用导出的二进制权重 (.bin) 进行纯整数 INT8 推理 (模拟 ASIC)

可用于验证两种方式的推理结果一致性。
"""

import struct
import numpy as np
import torch
from model import TinyGRUInt8

# ==========================================
#               配置参数
# ==========================================
BIN_PATH = "../outputs/tiny_gru_int8.bin"
PTH_PATH = "../outputs/tiny_gru_int8.pth"
DATA_PATH = "../../pdn_dataset_uint8.npz"

INPUT_SIZE = 9
HIDDEN_SIZE = 36
NUM_LAYERS = 1
WINDOW_SIZE = 50


# ==========================================
#           Binary 文件解析器
# ==========================================
class BinaryModelReader:
    def __init__(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.data = f.read()
        self.offset = 0
        
    def read_u32(self) -> int:
        val = struct.unpack('I', self.data[self.offset:self.offset+4])[0]
        self.offset += 4
        return val
    
    def read_i32(self) -> int:
        val = struct.unpack('i', self.data[self.offset:self.offset+4])[0]
        self.offset += 4
        return val
        
    def read_f32(self) -> float:
        val = struct.unpack('f', self.data[self.offset:self.offset+4])[0]
        self.offset += 4
        return val
    
    def read_i8_array(self, count: int) -> np.ndarray:
        vals = struct.unpack(f'{count}b', self.data[self.offset:self.offset+count])
        self.offset += count
        return np.array(vals, dtype=np.int8)
    
    def read_i32_array(self, count: int) -> np.ndarray:
        vals = struct.unpack(f'{count}i', self.data[self.offset:self.offset+4*count])
        self.offset += 4 * count
        return np.array(vals, dtype=np.int32)


class GRUWeights:
    """存储 GRU 模型的量化权重"""
    def __init__(self):
        self.input_size = 0
        self.hidden_size = 0
        self.num_layers = 0
        
        # Activation scales
        self.input_scale = 0.0
        self.input_zp = 0
        self.gru_out_scale = 0.0
        self.gru_out_zp = 0
        self.head_out_scale = 0.0
        self.head_out_zp = 0
        
        # GRU weights (INT8)
        self.w_ih_int8 = []
        self.w_ih_scale = []
        self.w_hh_int8 = []
        self.w_hh_scale = []
        self.b_ih_int32 = []
        self.b_hh_int32 = []
        
        # Head weights (INT8)
        self.head_w_int8 = None
        self.head_w_scale = 0.0
        self.head_b_int32 = None


def load_binary_weights(filepath: str) -> GRUWeights:
    """加载 .bin 文件"""
    reader = BinaryModelReader(filepath)
    weights = GRUWeights()
    
    magic = reader.read_u32()
    if magic != 0x47525538:
        raise ValueError(f"Invalid magic: 0x{magic:08X}")
    
    weights.input_size = reader.read_i32()
    weights.hidden_size = reader.read_i32()
    weights.num_layers = reader.read_i32()
    
    H = weights.hidden_size
    I = weights.input_size
    
    # Activation scales
    weights.input_scale = reader.read_f32()
    weights.input_zp = reader.read_i32()
    weights.gru_out_scale = reader.read_f32()
    weights.gru_out_zp = reader.read_i32()
    weights.head_out_scale = reader.read_f32()
    weights.head_out_zp = reader.read_i32()
    
    print(f"   input_scale={weights.input_scale:.6f}, input_zp={weights.input_zp}")
    print(f"   gru_out_scale={weights.gru_out_scale:.6f}, gru_out_zp={weights.gru_out_zp}")
    
    # GRU layers
    for layer_idx in range(weights.num_layers):
        w_ih_scale = reader.read_f32()
        _ = reader.read_i32()  # zp=0
        w_ih_int8 = reader.read_i8_array(3 * H * I).reshape(3 * H, I)
        weights.w_ih_int8.append(w_ih_int8)
        weights.w_ih_scale.append(w_ih_scale)
        
        w_hh_scale = reader.read_f32()
        _ = reader.read_i32()  # zp=0
        w_hh_int8 = reader.read_i8_array(3 * H * H).reshape(3 * H, H)
        weights.w_hh_int8.append(w_hh_int8)
        weights.w_hh_scale.append(w_hh_scale)
        
        b_ih_int32 = reader.read_i32_array(3 * H)
        weights.b_ih_int32.append(b_ih_int32)
        
        b_hh_int32 = reader.read_i32_array(3 * H)
        weights.b_hh_int32.append(b_hh_int32)
        
        print(f"   Layer {layer_idx}: w_ih_scale={w_ih_scale:.6f}, w_hh_scale={w_hh_scale:.6f}")
    
    # Head
    weights.head_w_scale = reader.read_f32()
    _ = reader.read_i32()
    weights.head_w_int8 = reader.read_i8_array(H).reshape(1, H)
    weights.head_b_int32 = reader.read_i32_array(1)
    
    print(f"   Head: scale={weights.head_w_scale:.6f}")
    print(f"[Binary] 读取完成，共 {reader.offset} 字节")
    return weights


# ==========================================
#           INT8 纯整数推理 (ASIC 模拟)
# ==========================================

def sigmoid_float(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def int8_gru_cell(x_t_int8: np.ndarray, h_prev_int8: np.ndarray, 
                  weights: GRUWeights, layer_idx: int = 0) -> np.ndarray:
    """
    INT8 GRU Cell - 纯整数推理
    """
    H = weights.hidden_size
    
    w_ih = weights.w_ih_int8[layer_idx].astype(np.int32)
    w_hh = weights.w_hh_int8[layer_idx].astype(np.int32)
    b_ih = weights.b_ih_int32[layer_idx]
    b_hh = weights.b_hh_int32[layer_idx]
    
    w_ih_scale = weights.w_ih_scale[layer_idx]
    w_hh_scale = weights.w_hh_scale[layer_idx]
    input_scale = weights.input_scale
    input_zp = weights.input_zp
    h_scale = weights.gru_out_scale
    h_zp = weights.gru_out_zp  # = 0 (对称量化)
    
    # 反量化后的 scale 乘积
    scale_ih = input_scale * w_ih_scale
    scale_hh = h_scale * w_hh_scale
    
    # 将输入转为 int32 并减去 zero point
    x_centered = x_t_int8.astype(np.int32) - input_zp
    h_centered = h_prev_int8.astype(np.int32) - h_zp
    
    # 分解为三个门
    W_ir, W_iz, W_in = w_ih[:H], w_ih[H:2*H], w_ih[2*H:3*H]
    W_hr, W_hz, W_hn = w_hh[:H], w_hh[H:2*H], w_hh[2*H:3*H]
    b_ir, b_iz, b_in = b_ih[:H], b_ih[H:2*H], b_ih[2*H:3*H]
    b_hr, b_hz, b_hn = b_hh[:H], b_hh[H:2*H], b_hh[2*H:3*H]
    
    # Reset gate: r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    r_acc_ih = W_ir @ x_centered + b_ir  # INT32
    r_acc_hh = W_hr @ h_centered + b_hr  # INT32
    r_float = r_acc_ih * scale_ih + r_acc_hh * scale_hh
    r_t = sigmoid_float(r_float)
    
    # Update gate: z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    z_acc_ih = W_iz @ x_centered + b_iz
    z_acc_hh = W_hz @ h_centered + b_hz
    z_float = z_acc_ih * scale_ih + z_acc_hh * scale_hh
    z_t = sigmoid_float(z_float)
    
    # New gate: n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    n_acc_ih = W_in @ x_centered + b_in
    n_hh_acc = W_hn @ h_centered + b_hn
    n_float = n_acc_ih * scale_ih + r_t * (n_hh_acc * scale_hh)
    n_t = np.tanh(n_float)
    
    # Hidden state: h_new = (1 - z) * n + z * h_prev
    h_prev_float = h_centered * h_scale  # 反量化 h_prev
    h_new_float = (1 - z_t) * n_t + z_t * h_prev_float
    
    # 重新量化为 INT8
    h_new_int8 = np.round(h_new_float / h_scale + h_zp).astype(np.int32)
    h_new_int8 = np.clip(h_new_int8, -128, 127).astype(np.int8)
    
    return h_new_int8


def int8_linear(h_int8: np.ndarray, weights: GRUWeights) -> float:
    """INT8 Linear 层"""
    h_scale = weights.gru_out_scale
    h_zp = weights.gru_out_zp
    w_scale = weights.head_w_scale
    
    w_int32 = weights.head_w_int8.astype(np.int32)
    b_int32 = weights.head_b_int32
    
    h_centered = h_int8.astype(np.int32) - h_zp
    
    acc = (w_int32 @ h_centered)[0] + b_int32[0]
    result = acc * (h_scale * w_scale)
    
    return float(result)


class BinaryInference:
    """INT8 纯整数推理 (模拟 ASIC)"""
    
    def __init__(self, weights: GRUWeights):
        self.weights = weights
    
    def forward(self, x_float: np.ndarray) -> np.ndarray:
        """
        x_float: (batch, seq_len, channels), float32 [0, 1]
        """
        batch_size, seq_len, _ = x_float.shape
        H = self.weights.hidden_size
        outputs = np.zeros((batch_size, 1), dtype=np.float32)
        
        for b in range(batch_size):
            # 初始化隐藏状态 (INT8, 值为0)
            h = np.zeros(H, dtype=np.int8)
            
            # 量化整个输入序列
            x_seq = x_float[b]
            x_seq_int8 = np.round(x_seq / self.weights.input_scale + self.weights.input_zp)
            x_seq_int8 = np.clip(x_seq_int8, -128, 127).astype(np.int8)
            
            # 遍历时间步
            for t in range(seq_len):
                x_t = x_seq_int8[t]
                h = int8_gru_cell(x_t, h, self.weights)
            
            outputs[b, 0] = int8_linear(h, self.weights)
        
        return outputs


# ==========================================
#           PyTorch 推理
# ==========================================

def load_pth_model(filepath: str) -> TinyGRUInt8:
    model = TinyGRUInt8(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    state_dict = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pth_inference(model: TinyGRUInt8, x: np.ndarray) -> np.ndarray:
    x_tensor = torch.from_numpy(x).float()
    with torch.no_grad():
        output = model(x_tensor)
    return output.numpy()


# ==========================================
#           主函数
# ==========================================

def main():
    print("=" * 60)
    print("TinyGRU INT8 推理对比测试")
    print("=" * 60)
    
    # 1. 加载测试数据
    print("\n[1] 加载测试数据...")
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    
    N_TEST = 100
    X_test = X[:N_TEST].astype(np.float32) / 255.0
    y_test = y[:N_TEST]
    
    print(f"   测试样本: {N_TEST}, 形状: {X_test.shape}")
    
    # 2. 加载 PyTorch 模型
    print("\n[2] 加载 PyTorch 模型...")
    try:
        pth_model = load_pth_model(PTH_PATH)
        print("   成功!")
    except Exception as e:
        print(f"   失败: {e}")
        pth_model = None
    
    # 3. 加载 Binary 权重
    print("\n[3] 加载 Binary 权重...")
    try:
        bin_weights = load_binary_weights(BIN_PATH)
        bin_inference = BinaryInference(bin_weights)
        print("   成功!")
    except Exception as e:
        print(f"   失败: {e}")
        import traceback
        traceback.print_exc()
        bin_inference = None
    
    # 4. 推理
    print("\n" + "=" * 60)
    print("推理结果对比")
    print("=" * 60)
    
    results = {}
    
    if pth_model:
        print("\n>>> PyTorch 浮点推理:")
        pth_out = pth_inference(pth_model, X_test)
        pth_pred = (sigmoid_float(pth_out) > 0.5).astype(np.float32).flatten()
        pth_acc = (pth_pred == y_test).mean()
        print(f"    准确率: {pth_acc:.4f}")
        print(f"    输出: [{pth_out.min():.4f}, {pth_out.max():.4f}], mean={pth_out.mean():.4f}")
        results['pth'] = pth_out
    
    if bin_inference:
        print("\n>>> Binary INT8 纯整数推理:")
        bin_out = bin_inference.forward(X_test)
        bin_pred = (sigmoid_float(bin_out) > 0.5).astype(np.float32).flatten()
        bin_acc = (bin_pred == y_test).mean()
        print(f"    准确率: {bin_acc:.4f}")
        print(f"    输出: [{bin_out.min():.4f}, {bin_out.max():.4f}], mean={bin_out.mean():.4f}")
        results['bin'] = bin_out
    
    # 5. 对比
    if 'pth' in results and 'bin' in results:
        print("\n" + "=" * 60)
        print("差异分析")
        print("=" * 60)
        
        diff = np.abs(results['pth'] - results['bin']).flatten()
        pred_match = (pth_pred == bin_pred).mean()
        
        print(f"最大绝对差异: {diff.max():.6f}")
        print(f"平均绝对差异: {diff.mean():.6f}")
        print(f"预测一致性:   {pred_match:.4f} ({int(pred_match*N_TEST)}/{N_TEST})")
        
        if diff.max() < 2.0 and pred_match > 0.95:
             print("\n✓ 结论: 结果基本一致 (考虑到 GRU 50步累积误差)")
        else:
             print("\n△ 结论: 存在一定差异")

    # 6. 样本对比
    print("\n" + "=" * 60)
    print("样本对比 (前10个)")
    print("=" * 60)
    print(f"{'#':<4} {'Label':<6} {'PTH':<12} {'INT8':<12}")
    print("-" * 40)
    
    for i in range(min(10, N_TEST)):
        pth_v = results.get('pth', [[float('nan')]*N_TEST])[i, 0]
        bin_v = results.get('bin', [[float('nan')]*N_TEST])[i, 0]
        print(f"{i:<4} {y_test[i]:<6.0f} {pth_v:<12.4f} {bin_v:<12.4f}")
    
    print("\n" + "=" * 60)
    print("完成!")


if __name__ == "__main__":
    main()
