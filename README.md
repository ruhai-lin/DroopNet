# DroopNet: ASIC-Oriented Temporal Neural Network Benchmark

## 📖 Introduction

**DroopNet** is a neural network benchmark project dedicated to processing temporal data under extreme hardware resource constraints. Our core objective is to optimize the most suitable neural network architecture for **Power Delivery Network (PDN)** voltage droop prediction, specifically targeted for deployment on ASICs with **only 8KB of SRAM**.

We selected **Temporal Convolutional Network (TCN)** as our primary architecture due to its superior parallelizability, deterministic memory access patterns, and excellent support in modern deep learning frameworks compared to RNNs (like GRU/LSTM). This project is inspired by **Proactive voltage droop mitigation techniques for high performance processors**, a concept detailed in the thesis by Jimmy Zhang (see [University of Illinois IDEALS](https://www.ideals.illinois.edu/items/126738)).

The project focuses on finding the optimal balance between accuracy, latency, and area (PPA), providing an intelligent core for next-generation high-performance power management chips.

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

To ensure efficient operation on low-cost ASICs, the TCN model strictly adheres to the following design specifications:

### 1. Memory Budget
- **Total SRAM**: 8KB (8192 Bytes).
- **Target Usage**: **< 6KB**.
  - *Note*: Approximately 2KB is reserved for system control logic, register configuration, and I/O buffering. The sum of model Weights, Biases, and runtime Activations (Ping-Pong Buffers) must not exceed 6KB.

### 2. Quantization Scheme
The model uses **W8A8** integer and fixed-point arithmetic, completely eliminating floating-point operations:
- **Strategy**: Post-Training Quantization (PTQ).
- **Weights**: **INT8** (Symmetric/Asymmetric quantization).
- **Bias**: **INT32** (To prevent accumulation overflow).
- **Activations**: **INT8** (Output of every layer is re-quantized to 8-bit).
- **Advantage**: Simplifies ASIC MAC unit design and significantly reduces power consumption.

### 3. Project Structure (TCN)
To achieve seamless delivery from algorithm to RTL, the architecture contains the following modules:

- **🐍 PythonModel**:
  - `model.py`: Defines the TCN architecture and simulates quantization effects (Fake Quantization).
  - `train.py`: Trains the network and exports float32 weights.
  - `inference.py`: Performs quantized inference, generating `.bin` weights and intermediate activation values as "Golden Vectors".

- **⚙️ CModel**:
  - `model.c/h`: Defines the `TinyTCNModel` structure and functions to load the binary weight file (`.bin`).
  - `inference.c/h`: The core engine. Implements standard INT8 operators (`Conv1D`, `Add`, `ReLU`, `Linear`) and the TCN block forward pass. It manages memory buffers to ensure zero dynamic allocation during inference.
  - `main.c`: The test harness. It loads the model and test data (`pdn_dataset_uint8.bin`), runs the inference loop, calculates accuracy/F1 scores, benchmarks latency, and verifies if the total memory footprint fits within the 8KB limit.

---

## 🚀 Quick Start

### 1. Environment Setup

It is recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

Run the physical emulator to generate training data and verification vectors:

```bash
python pdn_emulator.py
```
This will produce:
- `pdn_dataset_uint8.npz`: For Python training.
- `pdn_dataset_uint8.bin`: For C Model verification.

### 3. Train TCN Model

Train the model and export quantized weights:

```bash
cd TCN/PythonModel
python train.py
```
This saves the model weights to `../../outputs/tiny_tcn_int8.bin`.

### 4. Verify C Model

Compile and run the C reference implementation to verify accuracy and performance:

```bash
cd ../CModel
make
./tiny_tcn_test
```
*Check the output to ensure "8KB SRAM Fit" is YES and accuracy matches the Python model.*
