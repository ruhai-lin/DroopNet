import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import struct

# --- 1. 配置参数 (Configuration) ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024

# 频率设置
PHYSICS_FREQ = 1e9    # 1 GHz (物理模拟)
SENSOR_FREQ  = 5e6    # 5 MHz (传感器采样, 200ns/step)
NN_FREQ      = 5e6    # 神经网络频率

# 采样精度设置
ADC_BITS = 8
V_ADC_MIN = 1.0
V_ADC_MAX = 1.5
I_ADC_MIN = 0.0
I_ADC_MAX = 22.0      #稍微留点余量

# 物理参数
C_NODE   = 2.2e-6      
L_SERIES = 500e-12     
R_SERIES = 0.006       # <--- 关键修改: 增大电阻 (0.003->0.006). 
                       # 解释: 因为爬坡慢了，电感效应变弱，必须靠电阻压降来保证 20A 时电压一定跌落。
                       # 20A * 0.006 = 0.12V. 1.4 - 0.12 = 1.28V (< 1.30V Threshold). 稳!

V_SUPPLY = 1.4         
V_DROOP_TH = 1.30      

# 负载参数
I_BASE   = 5.0           
I_MEDIUM = 10.0        # 10A * 0.006 = 0.06V Drop -> 1.34V (Safe)
I_MAX    = 20.0        # 20A * 0.006 = 0.12V Drop -> 1.28V (Droop)

# 爬坡设置 (极慢爬坡，特征极其明显)
RAMP_TIME_US = 5.0     # <--- 关键修改: 5微秒的爬坡时间!
RAMP_STEPS_PHYS = int(RAMP_TIME_US * 1e-6 * PHYSICS_FREQ) # 5000 steps

SLEW_RATE = (I_MAX - I_BASE) / RAMP_STEPS_PHYS

# 状态机概率 (因为动作变慢了，概率要相应调低，避免重叠)
PROB_IDLE_TO_RAMP = 0.00002 
PROB_HOLD_TO_DOWN = 0.0002

# 时间设置 (延长总时间以收集足够样本)
TOTAL_TIME_US = 400.0 
STEPS = int(TOTAL_TIME_US * 1e-6 * PHYSICS_FREQ)

# 预测设置
PREDICTION_HORIZON_US = 2.0
HORIZON_STEPS = int(PREDICTION_HORIZON_US * 1e-6 * SENSOR_FREQ)

# 窗口设置 (给模型更长的记忆)
WINDOW_SIZE = 50       # <--- 关键修改: 50个点 (10us)，足够覆盖整个 5us 的爬坡过程

print(f"Running on {DEVICE}. Batch: {BATCH_SIZE}. Window: {WINDOW_SIZE}. Ramp: {RAMP_TIME_US}us")

# --- 2. 辅助函数 ---
@torch.jit.script
def val_to_adc_code(val: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    norm = (val - min_val) / (max_val - min_val)
    code = torch.clamp(torch.floor(norm * 255.0), 0.0, 255.0)
    return code

@torch.jit.script
def adc_code_to_val(code: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return code / 255.0 * (max_val - min_val) + min_val

# --- 3. 物理核心 ---
@torch.jit.script
def rlc_step(v_cap, i_ind, i_load,
             v_supply: float, r_series: float, l_series: float, c_node: float, dt: float):
    v_diff = v_supply - v_cap - (i_ind * r_series)
    d_i_ind = (v_diff / l_series) * dt
    new_i_ind = i_ind + d_i_ind
    i_net = new_i_ind - i_load
    d_v_cap = (i_net / c_node) * dt
    new_v_cap = v_cap + d_v_cap
    return new_v_cap, new_i_ind

# --- 4. 主流程 ---
def main():
    start_time = time.time()

    # Init
    v_steady_val = V_SUPPLY - (I_BASE * R_SERIES)
    v_cap = torch.ones((BATCH_SIZE, 3, 3), device=DEVICE) * v_steady_val
    i_ind = torch.ones((BATCH_SIZE, 3, 3), device=DEVICE) * I_BASE
    i_load_analog = torch.ones((BATCH_SIZE, 3, 3), device=DEVICE) * I_BASE
    
    # States: 0:IDLE, 1:RAMP_H, 2:HOLD_H, 3:DOWN, 4:RAMP_M, 5:HOLD_M
    load_state = torch.zeros((BATCH_SIZE, 3, 3), device=DEVICE, dtype=torch.long)

    physics_per_sensor = int(PHYSICS_FREQ / SENSOR_FREQ)
    dt = 1.0 / PHYSICS_FREQ
    total_sensor_steps = int(STEPS / physics_per_sensor)

    # Memory Pre-allocation
    sensor_history_I = torch.zeros((total_sensor_steps, BATCH_SIZE, 9), dtype=torch.uint8)
    sensor_history_V = torch.zeros((total_sensor_steps, BATCH_SIZE, 9), dtype=torch.uint8)

    print("Starting Simulation Loop...")
    sensor_idx = 0

    for step in range(STEPS):
        
        # --- Logic Update ---
        rand_probs = torch.rand((BATCH_SIZE, 3, 3), device=DEVICE)
        
        # IDLE -> RAMP (Split 50/50 between Heavy and Medium)
        mask_idle = (load_state == 0)
        trigger = mask_idle & (rand_probs < PROB_IDLE_TO_RAMP)
        
        is_heavy = torch.rand((BATCH_SIZE, 3, 3), device=DEVICE) > 0.5
        load_state[trigger & is_heavy] = 1       # To Heavy
        load_state[trigger & (~is_heavy)] = 4    # To Medium

        # RAMP -> HOLD logic
        # Heavy
        mask_ramp_h = (load_state == 1)
        done_ramp_h = mask_ramp_h & (i_load_analog >= I_MAX * 0.99)
        load_state[done_ramp_h] = 2
        # Medium
        mask_ramp_m = (load_state == 4)
        done_ramp_m = mask_ramp_m & (i_load_analog >= I_MEDIUM * 0.99)
        load_state[done_ramp_m] = 5

        # HOLD -> DOWN
        mask_hold = (load_state == 2) | (load_state == 5)
        stop_hold = mask_hold & (rand_probs < PROB_HOLD_TO_DOWN)
        load_state[stop_hold] = 3

        # DOWN -> IDLE
        mask_down = (load_state == 3)
        done_down = mask_down & (i_load_analog <= I_BASE * 1.01)
        load_state[done_down] = 0

        # --- Target Current ---
        target_I = torch.ones_like(i_load_analog) * I_BASE
        target_I[(load_state == 1) | (load_state == 2)] = I_MAX
        target_I[(load_state == 4) | (load_state == 5)] = I_MEDIUM

        # --- Slew Rate ---
        diff = target_I - i_load_analog
        step_change = torch.clamp(diff, -SLEW_RATE, SLEW_RATE)
        i_load_analog = i_load_analog + step_change

        # Noise
        noise = torch.randn_like(i_load_analog) * 0.02 # 减少一点噪声，让曲线更平滑易学
        i_load_analog = i_load_analog + noise
        i_load_analog = torch.clamp(i_load_analog, 0.0, I_MAX * 1.5)

        # --- Physics ---
        v_cap, i_ind = rlc_step(
            v_cap, i_ind, i_load_analog,
            V_SUPPLY, R_SERIES, L_SERIES, C_NODE, dt
        )

        # --- Sensor Sampling ---
        if step % physics_per_sensor == 0 and sensor_idx < total_sensor_steps:
            i_load_code_float = val_to_adc_code(i_load_analog, I_ADC_MIN, I_ADC_MAX)
            sensor_history_I[sensor_idx] = i_load_code_float.view(BATCH_SIZE, -1).cpu().to(torch.uint8)
            
            v_code_float = val_to_adc_code(v_cap, V_ADC_MIN, V_ADC_MAX)
            sensor_history_V[sensor_idx] = v_code_float.view(BATCH_SIZE, -1).cpu().to(torch.uint8)
            
            sensor_idx += 1

    print(f"Simulation finished. {STEPS * BATCH_SIZE / (time.time() - start_time) / 1e6:.2f} M-Steps/s")

    # --- 5. 可视化检查 ---
    DROOP_CODE = int((V_DROOP_TH - V_ADC_MIN) / (V_ADC_MAX - V_ADC_MIN) * 255)
    
    # 查找包含 Droop 的 Batch
    min_codes = sensor_history_V.min(dim=0)[0].min(dim=1)[0]
    droop_batches = torch.where(min_codes < DROOP_CODE)[0]
    
    target_batch = droop_batches[0].item() if len(droop_batches) > 0 else 0
    print(f"Visualizing Batch #{target_batch}")

    trace_I = adc_code_to_val(sensor_history_I[:, target_batch, :].float(), I_ADC_MIN, I_ADC_MAX).numpy()
    trace_V = adc_code_to_val(sensor_history_V[:, target_batch, :].float(), V_ADC_MIN, V_ADC_MAX).numpy()
    t_axis = np.arange(len(trace_I)) * (1e6 / SENSOR_FREQ)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(f"Long Ramp (5us) & High Res (R=6mOhm)\nEasier to distinguish Heavy (20A) vs Medium (10A)", fontsize=14)

    for i in range(9):
        ax = axes[i//3, i%3]
        ax.plot(t_axis, trace_V[:, i], 'b', label='V', linewidth=1)
        ax.axhline(V_DROOP_TH, c='r', ls='--', alpha=0.5)
        
        ax2 = ax.twinx()
        ax2.plot(t_axis, trace_I[:, i], 'orange', label='I', alpha=0.8, linewidth=1.5)
        ax2.set_ylim(0, 24)
        
        # 标记 Droop
        is_droop = trace_V[:, i] < V_DROOP_TH
        ax.fill_between(t_axis, V_ADC_MIN, V_ADC_MAX, where=is_droop, color='red', alpha=0.2, transform=ax.get_xaxis_transform())
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 6. 生成宽窗口数据集 ---
    print(f"Generating Dataset with Window Size = {WINDOW_SIZE}...")
    
    raw_I = sensor_history_I.permute(1, 0, 2) # [Batch, Time, 9]
    raw_V = sensor_history_V.permute(1, 0, 2)
    
    global_droop_mask = (raw_V < DROOP_CODE).any(dim=2).float() # [Batch, Time]

    step_size = 1
    
    # 展开窗口
    X_windows = raw_I.unfold(1, WINDOW_SIZE, step_size)
    num_windows = X_windows.shape[1]
    
    # 构建标签 (预测未来)
    y_labels = torch.zeros((BATCH_SIZE, num_windows), dtype=torch.float32)
    
    print(f"Labelling {num_windows} windows...")
    for i in range(num_windows):
        t_now = i * step_size + WINDOW_SIZE
        t_future = t_now + HORIZON_STEPS
        
        if t_future >= global_droop_mask.shape[1]:
            X_windows = X_windows[:, :i]
            y_labels = y_labels[:, :i]
            break
            
        future_val = global_droop_mask[:, t_now:t_future].max(dim=1)[0]
        y_labels[:, i] = future_val

    X_final = X_windows.reshape(-1, WINDOW_SIZE, 9)
    y_final = y_labels.reshape(-1)

    print("-" * 30)
    print(f"Pos: {(y_final==1).sum()} | Neg: {(y_final==0).sum()}")
    print(f"Shape: {X_final.shape}")
    print("-" * 30)
    
    if (y_final==1).sum() > 0:
        # --- 7.1 导出 ASIC Verification Binary (.bin) ---
        # 格式: [Header] + 循环([Data X] + [Label y])
        print("Exporting Verification Vectors for C Model...")
        
        # 1. 转换为 Numpy
        X_numpy = X_final.cpu().numpy() # uint8
        y_numpy = y_final.cpu().numpy() # float
        
        # 2. 筛选 50 Pos / 50 Neg
        pos_indices = np.where(y_numpy == 1)[0]
        neg_indices = np.where(y_numpy == 0)[0]
        
        N_POS = 50
        N_NEG = 50
        
        # 防止样本不够
        real_n_pos = min(len(pos_indices), N_POS)
        real_n_neg = min(len(neg_indices), N_NEG)
        
        # 随机抽取
        sel_pos = np.random.choice(pos_indices, real_n_pos, replace=False)
        sel_neg = np.random.choice(neg_indices, real_n_neg, replace=False)
        
        # 合并并打乱
        indices = np.concatenate([sel_pos, sel_neg])
        np.random.shuffle(indices)
        
        X_subset = X_numpy[indices]      # uint8
        y_subset = y_numpy[indices].astype(np.uint8) # float -> uint8 (重要!)
        
        output_bin = "pdn_dataset_uint8.bin"
        total_samples = len(indices)
        channels = 9
        
        # 3. 写入二进制文件
        with open(output_bin, 'wb') as f:
            # Header: Magic(0xAABBCCDD), N, W, C
            header = struct.pack('Iiii', 0xAABBCCDD, total_samples, WINDOW_SIZE, channels)
            f.write(header)
            
            # Data Loop: X then y
            for i in range(total_samples):
                f.write(X_subset[i].tobytes()) # 写入 50*9 bytes
                f.write(y_subset[i].tobytes()) # 写入 1 byte
                
        print(f"Done! Saved {total_samples} samples to '{output_bin}'")
        np.savez("pdn_dataset_uint8.npz", X=X_final.cpu().numpy(), y=y_final.cpu().numpy())
        print("Saved pdn_dataset_uint8.npz")

if __name__ == "__main__":
    main()