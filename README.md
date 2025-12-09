# DroopNet: ASIC-Oriented Temporal Neural Network Benchmark

## 📖 Introduction

**DroopNet** is a neural network benchmark project dedicated to processing temporal data under extreme hardware resource constraints. Our core objective is to identify and optimize the most suitable neural network architectures for **Power Delivery Network (PDN)** voltage droop prediction, specifically targeted for deployment on ASICs with **only 8KB of SRAM**.

This project implements **Proactive voltage droop mitigation techniques for high performance processors**, a concept detailed in the thesis by Jimmy Zhang (see [University of Illinois IDEALS](https://www.ideals.illinois.edu/items/126738)).

By comparing various temporal architectures (such as GRU, TCN, etc.), we aim to find the optimal balance between accuracy, latency, and area (PPA), providing an intelligent core for next-generation high-performance power management chips.

---

## ⚡ Data Specification

The data is generated from a high-fidelity RLC physical simulation environment (`pdn_emulator.py`), which models the current and voltage response of a chip under drastic load changes.

### Input Features
- **Physical Meaning**: Load current sensor readings ($I_{load}$).
- **Channels**: 9 (Representing 9 different power domains/observation points on the chip).
- **Data Format**: 8-bit ADC values (UINT8, 0-255).
- **Window Size**: 50 Timesteps.
- **Sampling Frequency**: 5 MHz (Covering approximately 10$\mu s$ of history).
- **Input Tensor Shape**: `[Batch, 50, 9]`.

### Prediction Target
- **Task Type**: Binary Classification.
- **Definition**: Predict whether the voltage will drop below the threshold ($V_{th} = 1.30V$) within the next 2.0 $\mu s$ (10 timesteps).
- **Labels**: `1` (Droop Event), `0` (Safe).

---

## 🛠️ Hardware & Model Constraints

To ensure efficient operation on low-cost ASICs, all models must strictly adhere to the following design specifications:

### 1. Memory Budget
- **Total SRAM**: 8KB (8192 Bytes).
- **Target Usage**: **< 6KB**.
  - *Note*: Approximately 2KB is reserved for system control logic, register configuration, and I/O buffering. The sum of model Weights, Biases, and runtime Activations (Buffers) must not exceed 6KB.

### 2. Quantization Scheme
All models use **W8A8** integer and fixed-point arithmetic, completely eliminating floating-point operations:
- **Strategy**: Post-Training Quantization (PTQ).
- **Weights**: **INT8** (Symmetric/Asymmetric quantization).
- **Bias**: **INT32** (To prevent accumulation overflow).
- **Activations**: **INT8** (Output of every layer is re-quantized to 8-bit).
- **Advantage**: Simplifies ASIC MAC unit design and significantly reduces power consumption.

### 3. Project Structure
To achieve seamless delivery from algorithm to RTL, each network architecture (e.g., `GRU/`, `TCN/`) contains the following modules:

- **🐍 PythonModel**:
  - `model.py`: Defines the floating-point model and Fake Quantization logic.
  - `train.py`: Trains the network and exports float32 weights.
  - `inference.py`: Performs quantized inference, generating `.bin` weights and intermediate activation values as "Golden Vectors".

- **⚙️ CModel**:
  - `inference.c`: Pure C implementation of the fixed-point inference engine.
  - **Purpose**: Strictly mimics hardware behavior to verify quantization accuracy and serves as a behavioral reference for RTL design.

---

## 🚀 Quick Start

1. **Generate Data**:
   ```bash
   python pdn_emulator.py
   ```
   This generates training data `pdn_dataset_uint8.npz` and `pdn_dataset_uint8.bin` for C model verification.

2. **Train Model (e.g., GRU)**:
   ```bash
   cd GRU/PythonModel
   python train.py
   ```

3. **Verify C Model**:
   ```bash
   cd GRU/CModel
   make
   ./tiny_gru_test
   ```
