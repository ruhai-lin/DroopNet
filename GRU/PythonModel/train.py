"""
TinyGRU 训练脚本 - 训练 + 校准 + INT8 导出

针对 ASIC 部署:
1. 训练浮点模型
2. 校准激活值范围 (计算 Scale/ZP)
3. 导出 INT8 权重二进制文件 (bias 转 INT32)
"""

import copy
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import TinyGRUInt8

# ==========================================
#               Configuration
# ==========================================
DATA_PATH = "../../pdn_dataset_uint8.npz"
BATCH_SIZE = 2048
EPOCHS = 5
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
NUM_WORKERS = 4

HIDDEN_SIZE = 36
INPUT_SIZE = 9
SEQ_LEN = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
#               Dataset
# ==========================================
class PDNDataset(Dataset):
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        print(f"[Dataset] Loading from {data_path}...")
        data = np.load(data_path)
        self.X = data["X"]
        self.y = data["y"].astype(np.float32)
        print(f"[Dataset] Loaded. Shape: {self.X.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = torch.from_numpy(self.X[idx]).float() / 255.0
        y_sample = torch.tensor(self.y[idx])
        return x_sample, y_sample


# ==========================================
#          Calibration & Quantization
# ==========================================
class ActivationCalibrator:
    """收集激活值统计信息，计算量化参数"""
    
    def __init__(self):
        self.min_vals = {}
        self.max_vals = {}
    
    def reset(self):
        self.min_vals = {}
        self.max_vals = {}
    
    def update(self, name: str, tensor: torch.Tensor):
        t = tensor.detach()
        if name not in self.min_vals:
            self.min_vals[name] = t.min().item()
            self.max_vals[name] = t.max().item()
        else:
            self.min_vals[name] = min(self.min_vals[name], t.min().item())
            self.max_vals[name] = max(self.max_vals[name], t.max().item())
    
    def compute_scale_zp(self, name: str, symmetric: bool = True):
        """计算量化参数 (Signed INT8: -128 to 127)"""
        min_val = self.min_vals.get(name, 0)
        max_val = self.max_vals.get(name, 1)
        
        if symmetric:
            # Symmetric: max(|min|, |max|) -> 127
            abs_max = max(abs(min_val), abs(max_val))
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            zp = 0
        else:
            # Asymmetric: min->-128, max->127
            # Range = 255
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            # 0 = (val - min) / scale - 128
            # zp such that: quant = val/scale + zp
            # -128 = min/scale + zp  =>  zp = -128 - min/scale
            zp = int(round(-128 - min_val / scale))
            zp = max(-128, min(127, zp))
        
        return scale, zp


def calibrate_model(model: nn.Module, loader: DataLoader, device) -> ActivationCalibrator:
    """校准模型，收集激活值范围"""
    print("[Calibration] Collecting activation statistics...")
    
    model.eval()
    calibrator = ActivationCalibrator()
    
    with torch.no_grad():
        for X, _ in tqdm(loader, desc="Calibrate", leave=False):
            X = X.to(device)
            
            # 记录输入
            calibrator.update("input", X)
            
            # GRU 前向
            output, _ = model.gru(X)
            calibrator.update("gru_output", output)
            
            # 最后一个时间步
            last_output = output[:, -1, :]
            calibrator.update("head_input", last_output)
            
            # Head 输出
            logits = model.head(last_output)
            calibrator.update("head_output", logits)
    
    print("[Calibration] Done.")
    return calibrator


def quantize_weight(tensor: torch.Tensor) -> tuple:
    """量化权重为 INT8"""
    t = tensor.detach().cpu()
    abs_max = t.abs().max().item()
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    zp = 0
    
    quantized = torch.clamp(torch.round(t / scale), -128, 127)
    return quantized.numpy().astype(np.int8), scale, zp


def quantize_bias(bias: torch.Tensor, input_scale: float, weight_scale: float) -> np.ndarray:
    """量化 Bias 为 INT32"""
    b = bias.detach().cpu().numpy()
    scale = input_scale * weight_scale
    if scale == 0:
        scale = 1e-9
    b_int32 = np.round(b / scale).astype(np.int32)
    return b_int32


# ==========================================
#          Binary Export
# ==========================================
def export_gru_binary(model: nn.Module, calibrator: ActivationCalibrator, filepath: str):
    """
    导出 GRU 模型为二进制格式
    
    文件格式:
    [Header]
      - Magic: 0x47525538 ('GRU8')
      - input_size, hidden_size, num_layers (int32 x 3)
    
    [Activation Scales]
      - input_scale, input_zp
      - gru_output_scale, gru_output_zp
      - head_output_scale, head_output_zp
    
    [GRU Layer]
      - weight_ih: scale, zp, data (int8)
      - weight_hh: scale, zp, data (int8)
      - bias_ih: data (int32)
      - bias_hh: data (int32)
    
    [Head Layer]
      - weight: scale, zp, data (int8)
      - bias: data (int32)
    """
    print(f"\n[Export] Saving GRU binary to {filepath}...")
    
    model = model.cpu().eval()
    info = model.get_export_info()
    H = info['hidden_size']
    I = info['input_size']
    
    # 获取激活值量化参数
    input_scale, input_zp = calibrator.compute_scale_zp("input", symmetric=False)
    gru_out_scale, gru_out_zp = calibrator.compute_scale_zp("gru_output", symmetric=True)
    head_out_scale, head_out_zp = calibrator.compute_scale_zp("head_output", symmetric=True)
    
    with open(filepath, 'wb') as f:
        # Header
        f.write(struct.pack('I', 0x47525538))  # Magic
        f.write(struct.pack('iii', I, H, info['num_layers']))
        
        # Activation Scales
        f.write(struct.pack('fi', input_scale, input_zp))
        f.write(struct.pack('fi', gru_out_scale, gru_out_zp))
        f.write(struct.pack('fi', head_out_scale, head_out_zp))
        
        # GRU Layer
        gru = model.gru
        for layer_idx in range(info['num_layers']):
            suffix = f'_l{layer_idx}'
            
            # weight_ih: (3*H, I)
            w_ih = getattr(gru, f'weight_ih{suffix}')
            w_ih_int8, w_ih_scale, w_ih_zp = quantize_weight(w_ih)
            f.write(struct.pack('fi', w_ih_scale, w_ih_zp))
            f.write(w_ih_int8.tobytes())
            
            # weight_hh: (3*H, H)
            w_hh = getattr(gru, f'weight_hh{suffix}')
            w_hh_int8, w_hh_scale, w_hh_zp = quantize_weight(w_hh)
            f.write(struct.pack('fi', w_hh_scale, w_hh_zp))
            f.write(w_hh_int8.tobytes())
            
            # bias_ih: (3*H,) -> INT32
            b_ih = getattr(gru, f'bias_ih{suffix}')
            b_ih_int32 = quantize_bias(b_ih, input_scale, w_ih_scale)
            f.write(b_ih_int32.tobytes())
            
            # bias_hh: (3*H,) -> INT32
            b_hh = getattr(gru, f'bias_hh{suffix}')
            b_hh_int32 = quantize_bias(b_hh, gru_out_scale, w_hh_scale)
            f.write(b_hh_int32.tobytes())
            
            print(f"  Layer {layer_idx}: w_ih_scale={w_ih_scale:.6f}, w_hh_scale={w_hh_scale:.6f}")
        
        # Head Layer
        head_w = model.head.weight
        head_w_int8, head_w_scale, head_w_zp = quantize_weight(head_w)
        f.write(struct.pack('fi', head_w_scale, head_w_zp))
        f.write(head_w_int8.tobytes())
        
        head_b = model.head.bias
        head_b_int32 = quantize_bias(head_b, gru_out_scale, head_w_scale)
        f.write(head_b_int32.tobytes())
        
        print(f"  Head: scale={head_w_scale:.6f}")
    
    file_size = os.path.getsize(filepath)
    print(f"[Export] Done. File size: {file_size} bytes ({file_size/1024:.2f} KB)")


# ==========================================
#          Training & Evaluation
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, desc="Train", leave=False)
    for X, y in loop:
        X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device).unsqueeze(1)
        outputs = model(X)
        loss = criterion(outputs, y)
        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    return {
        "loss": total_loss / max(1, len(loader)),
        "acc": accuracy_score(all_targets, all_preds),
        "prec": precision_score(all_targets, all_preds, zero_division=0),
        "rec": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0)
    }


def prepare_dataloaders():
    full_dataset = PDNDataset(DATA_PATH)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader


# ==========================================
#               Main Function
# ==========================================
def main():
    print("-" * 60)
    print(f"TinyGRU Training | Device: {DEVICE}")
    print("-" * 60)

    # 1. Setup Data & Model
    train_loader, val_loader = prepare_dataloaders()
    
    model = TinyGRUInt8(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    model = model.to(DEVICE)
    
    print(f"Model Estimate (Float): {model.footprint_bytes()} bytes")

    # 2. Setup Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_f1 = 0.0
    best_state = None

    # 3. Training Loop
    for epoch in range(EPOCHS):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        metrics = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {t_loss:.4f} | Val Loss: {metrics['loss']:.4f} | "
            f"F1: {metrics['f1']:.4f} (Acc: {metrics['acc']:.4f})"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

    # 4. Finalizing Float Model
    if best_state:
        print(f"\nLoading best model (F1: {best_f1:.4f})...")
        model.load_state_dict(best_state)
    
    torch.save(model.state_dict(), "../outputs/tiny_gru_float.pth")
    print("Saved float model to ../outputs/tiny_gru_float.pth")

    # 5. Calibration (相当于 PTQ 校准)
    print("-" * 60)
    print("Starting Calibration...")
    print("-" * 60)
    
    model.cpu().eval()
    calibrator = calibrate_model(model, val_loader, device=torch.device("cpu"))

    # 6. 导出文件
    os.makedirs("../outputs", exist_ok=True)
    
    # A. PyTorch 格式 (用于 Python 推理)
    torch.save(model.state_dict(), "../outputs/tiny_gru_int8.pth")
    print("Saved INT8 model to ../outputs/tiny_gru_int8.pth")
    
    # B. 嵌入式格式 (Raw Binary)
    export_gru_binary(model, calibrator, "../outputs/tiny_gru_int8.bin")

    print("-" * 60)
    print("All Done.")


if __name__ == "__main__":
    main()
