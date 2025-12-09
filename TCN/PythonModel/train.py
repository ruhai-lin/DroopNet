import copy
import os
import numpy as np
import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# 假设 model.py 在同级目录下
from model import TinyTCNInt8

# ==========================================
#               Configuration
# ==========================================
DATA_PATH = "../../pdn_dataset_uint8.npz"
BATCH_SIZE = 2048
EPOCHS = 5
LEARNING_RATE = 2e-3
VAL_SPLIT = 0.2
NUM_WORKERS = 4

# PTQ Settings
Q_ENGINE = "fbgemm"
CALIBRATION_BATCHES = 100  # 用于校准的批次数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.quantized.engine = Q_ENGINE

# ==========================================
#               Dataset
# ==========================================
class PDNDataset(Dataset):
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        print(f"[Dataset] Loading from {data_path}...")
        try:
            data = np.load(data_path)
            self.X = data["X"]  # Expecting uint8
            self.y = data["y"].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        print(f"[Dataset] Loaded. Shape: {self.X.shape}")
        if self.X.shape[1] < 50:
            print("[Warning] Window size < 50, check TCN receptive field requirements.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = torch.from_numpy(self.X[idx]).float() / 255.0  # uint8 -> float [0, 1]
        y_sample = torch.tensor(self.y[idx])      # float32 -> tensor
        return x_sample, y_sample

# ==========================================
#          Helper: Flat Binary Export
# ==========================================
def write_layer_weights(f, module, input_scale, input_zp):
    """
    Writes a single quantized layer (Conv1d or Linear) to file.
    Format:
      [Input Scale (f32), Input ZP (i32), Output Scale (f32), Output ZP (i32)]
      [Scales (f32 array), ZPs (i32 array), Bias (i32 array), Weights (i8 array)]
    """
    # 1. Get Weight Packing
    w_packed = module.weight()
    n_channels = w_packed.shape[0] # Out channels
    
    # 2. Get Weight Scales/ZPs (Per-Channel or Per-Tensor)
    if w_packed.qscheme() in [torch.per_channel_affine, torch.per_channel_symmetric]:
        w_scales = w_packed.q_per_channel_scales().numpy().astype(np.float32)
        w_zps = w_packed.q_per_channel_zero_points().numpy().astype(np.int32)
    else:
        s = float(w_packed.q_scale())
        z = int(w_packed.q_zero_point())
        w_scales = np.full(n_channels, s, dtype=np.float32)
        w_zps = np.full(n_channels, z, dtype=np.int32)
    
    # 3. Get Weights (int8)
    w_int8 = w_packed.int_repr().cpu().numpy().astype(np.int8)

    # 4. Get Output Scale/ZP
    out_scale = float(module.scale)
    out_zp = int(module.zero_point)
    
    # 5. Get Bias and Quantize to Int32
    # Bias is float. We need to quantize it as: bias_int = round(bias_float / (input_scale * w_scales))
    # Note: w_scales is an array. input_scale is scalar.
    bias = module.bias()
    if bias is None:
        b_int32 = np.zeros(n_channels, dtype=np.int32)
    else:
        b_fp32 = bias.detach().cpu().numpy()
        # Quantize
        scale_factor = input_scale * w_scales
        # Avoid division by zero
        scale_factor[scale_factor == 0] = 1e-9
        b_int32 = np.round(b_fp32 / scale_factor).astype(np.int32)

    # 6. Write to File
    # A. Quantization Params (Input/Output)
    f.write(np.array([input_scale], dtype=np.float32).tobytes())
    f.write(np.array([input_zp], dtype=np.int32).tobytes())
    f.write(np.array([out_scale], dtype=np.float32).tobytes())
    f.write(np.array([out_zp], dtype=np.int32).tobytes())
    
    # B. Layer Arrays
    f.write(w_scales.tobytes())
    f.write(w_zps.tobytes())
    f.write(b_int32.tobytes())
    f.write(w_int8.tobytes())
    
    return out_scale, out_zp # Return output stats for next layer

def export_flat_binary(model: nn.Module, filepath: str):
    """
    Exports TinyTCNInt8 to binary for C inference.
    """
    print(f"\n[Export] Saving raw binary to {filepath}...")
    
    with open(filepath, "wb") as f:
        # Initial Input Scale (from QuantStub)
        current_scale = float(model.quant.scale)
        current_zp = int(model.quant.zero_point)
        
        # Iterate Blocks
        for i, block in enumerate(model.network):
            print(f"  Block {i}...")
            # Conv1
            mid_scale, mid_zp = write_layer_weights(f, block.conv1, current_scale, current_zp)
            
            # Conv2 (Input is Conv1 Output)
            write_layer_weights(f, block.conv2, mid_scale, mid_zp)
            
            # Downsample (Input is Block Input)
            if block.downsample is not None:
                 write_layer_weights(f, block.downsample, current_scale, current_zp)
            
            # Block Output (skip_add)
            # Check if skip_add is QFunctional and has scale/zp
            # If not (e.g. FloatFunctional wasn't converted properly), we have a problem.
            # Assuming 'convert' turned FloatFunctional into QFunctional
            if hasattr(block.skip_add, 'scale'):
                block_out_scale = float(block.skip_add.scale)
                block_out_zp = int(block.skip_add.zero_point)
            else:
                # Fallback: if not observed, maybe identity? 
                # But PTQ usually observes add.
                print(f"    [Warning] Block {i} skip_add has no scale/zp. Using input stats.")
                block_out_scale = current_scale
                block_out_zp = current_zp
            
            # Write Block Output Scale/ZP (for residual add in C)
            f.write(np.array([block_out_scale], dtype=np.float32).tobytes())
            f.write(np.array([block_out_zp], dtype=np.int32).tobytes())
            
            # Update current scale for next block
            current_scale = block_out_scale
            current_zp = block_out_zp

        # Head
        print("  Head...")
        write_layer_weights(f, model.head, current_scale, current_zp)

    print(f"[Export] Done.")


# ==========================================
#          Training & Evaluation Logic
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

    metrics = {
        "loss": total_loss / max(1, len(loader)),
        "acc": accuracy_score(all_targets, all_preds),
        "prec": precision_score(all_targets, all_preds, zero_division=0),
        "rec": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0)
    }
    return metrics


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


@torch.no_grad()
def calibrate_model(model, loader, num_batches):
    """
    使用校准数据集收集量化统计信息 (PTQ 核心步骤)
    """
    model.eval()
    print(f"[PTQ] Calibrating with {num_batches} batches...")
    
    for i, (X, _) in enumerate(loader):
        if i >= num_batches:
            break
        X = X.to(torch.device("cpu"))  # PTQ 校准在 CPU 上进行
        model(X)
        
        if (i + 1) % 10 == 0:
            print(f"    Calibrated {i + 1}/{num_batches} batches")
    
    print("[PTQ] Calibration done.")


# ==========================================
#               Main Function
# ==========================================
def main():
    print("-" * 60)
    print(f"TinyTCN PTQ Training | Device: {DEVICE}")
    print("-" * 60)

    # 1. Setup Data & Model (普通浮点训练，不使用 QAT)
    train_loader, val_loader = prepare_dataloaders()
    
    model = TinyTCNInt8()
    model = model.to(DEVICE)
    
    print(f"Model Estimate (Float): {model.footprint_bytes()} bytes")

    # 2. Setup Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=2)

    best_f1 = 0.0
    best_state = None

    # 3. Training Loop (普通浮点训练)
    for epoch in range(EPOCHS):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        metrics = evaluate(model, val_loader, criterion)
        scheduler.step(metrics["f1"])

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
    
    # 保存浮点模型 checkpoint
    torch.save(model.state_dict(), "../outputs/tiny_tcn_float.pth")
    print("Saved float model to ../outputs/tiny_tcn_float.pth")

    # 5. PTQ: Post-Training Quantization
    print("-" * 60)
    print("Starting Post-Training Quantization (PTQ)...")
    print("-" * 60)
    
    # 将模型移至 CPU 进行 PTQ
    model.cpu().eval()
    
    # 5.1 Fuse Modules (Conv + ReLU)
    print("[PTQ] Fusing modules...")
    model.fuse_model()
    
    # 5.2 设置量化配置
    print("[PTQ] Setting up quantization config...")
    model.qconfig = tq.get_default_qconfig(Q_ENGINE)
    
    # 5.3 Prepare for PTQ (插入 Observer，但不是伪量化节点)
    print("[PTQ] Preparing model for quantization...")
    tq.prepare(model, inplace=True)
    
    # 5.4 校准：使用训练数据收集统计信息
    print("[PTQ] Running calibration...")
    calibrate_model(model, train_loader, CALIBRATION_BATCHES)
    
    # 5.5 转换为真正的 INT8 模型
    print("[PTQ] Converting to INT8 model...")
    tq.convert(model, inplace=True)

    # 6. 验证 INT8 模型的精度
    print("-" * 60)
    print("Validating INT8 Model on CPU...")
    q_metrics = evaluate(model, val_loader, criterion, device=torch.device("cpu"))
    print(f"Final INT8 F1: {q_metrics['f1']:.4f} | Acc: {q_metrics['acc']:.4f}")

    # 7. 导出文件
    # A. PyTorch 格式 (用于 Python 推理)
    torch.save(model.state_dict(), "../outputs/tiny_tcn_int8.pth")
    print("Saved INT8 model to ../outputs/tiny_tcn_int8.pth")
    
    # B. 嵌入式格式 (Raw Binary, Small Size!)
    export_flat_binary(model, "../outputs/tiny_tcn_int8.bin")

    print("-" * 60)
    print("All Done.")


if __name__ == "__main__":
    main()

