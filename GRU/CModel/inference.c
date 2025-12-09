/**
 * @file inference.c
 * @brief TinyGRU INT8 inference implementation - ASIC Golden Model
 */

#include "inference.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================
 *          Buffer management
 * ======================================== */

int inference_buffer_init(InferenceBuffer *buf, int hidden_size) {
    buf->h_state = (int8_t *)calloc(hidden_size, sizeof(int8_t));
    buf->acc_buffer = (int32_t *)calloc(3 * hidden_size, sizeof(int32_t));
    buf->gate_buffer = (float *)calloc(3 * hidden_size, sizeof(float));
    
    if (!buf->h_state || !buf->acc_buffer || !buf->gate_buffer) {
        inference_buffer_free(buf);
        return -1;
    }
    
    return 0;
}

void inference_buffer_free(InferenceBuffer *buf) {
    if (buf->h_state) { free(buf->h_state); buf->h_state = NULL; }
    if (buf->acc_buffer) { free(buf->acc_buffer); buf->acc_buffer = NULL; }
    if (buf->gate_buffer) { free(buf->gate_buffer); buf->gate_buffer = NULL; }
}

void inference_buffer_reset(InferenceBuffer *buf, int hidden_size) {
    memset(buf->h_state, 0, hidden_size * sizeof(int8_t));
}

/* ========================================
 *          Quantization basics
 * ======================================== */

int8_t quantize_int8(float val, float scale, int32_t zp) {
    int32_t q = (int32_t)roundf(val / scale) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

float dequantize_int8(int8_t val, float scale, int32_t zp) {
    return ((float)val - (float)zp) * scale;
}

/* ========================================
 *          Activation functions
 * ======================================== */

static float sigmoid_f(float x) {
    // Prevent overflow
    if (x < -50.0f) return 0.0f;
    if (x > 50.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float tanh_f(float x) {
    return tanhf(x);
}

/* ========================================
 *          GRU cell implementation
 * ======================================== */

/**
 * @brief INT8 matrix-vector multiply + bias
 * 
 * Compute: out = W @ x + b (quantized)
 * W: [out_size, in_size], int8
 * x: [in_size], int8
 * b: [out_size], int32
 * 
 * Returns dequantized float result
 */
static void matmul_int8(float *out,
                        const int8_t *W, const int8_t *x,
                        const int32_t *bias,
                        int out_size, int in_size,
                        float w_scale, float x_scale,
                        int32_t x_zp) {
    
    float scale = w_scale * x_scale;
    
    for (int i = 0; i < out_size; i++) {
        int32_t acc = 0;
        
        // INT8 matmul
        for (int j = 0; j < in_size; j++) {
            // Note: input needs zero-point subtraction
            acc += (int32_t)W[i * in_size + j] * ((int32_t)x[j] - x_zp);
        }
        
        // Add bias
        acc += bias[i];
        
        // Dequantize
        out[i] = (float)acc * scale;
    }
}

void gru_cell(int8_t *h_out,
              const int8_t *h_prev,
              const int8_t *x_t,
              const TinyGRUModel *model,
              InferenceBuffer *buf,
              int layer_idx) {
    
    const GRULayerParams *gru = &model->gru_layers[layer_idx];
    int H = model->hidden_size;
    int I = model->input_size;
    
    // Gate computation buffers
    float *gates_ih = buf->gate_buffer;           // [3*H] for W_ih @ x
    float *gates_hh = buf->gate_buffer;           // reused, computed in steps
    
    // Temporary storage
    float r_ih[HIDDEN_SIZE], z_ih[HIDDEN_SIZE], n_ih[HIDDEN_SIZE];
    float r_hh[HIDDEN_SIZE], z_hh[HIDDEN_SIZE], n_hh[HIDDEN_SIZE];
    
    // 1. Compute W_ih @ x_t + b_ih
    // Split into three gates: r, z, n
    // Input x_t needs zero-point subtraction
    matmul_int8(gates_ih, gru->w_ih_int8, x_t, gru->b_ih_int32,
                3 * H, I, gru->w_ih_scale, model->input_scale, model->input_zp);
    
    for (int i = 0; i < H; i++) {
        r_ih[i] = gates_ih[i];
        z_ih[i] = gates_ih[H + i];
        n_ih[i] = gates_ih[2 * H + i];
    }
    
    // 2. Compute W_hh @ h_{t-1} + b_hh
    // Hidden state h_prev is symmetric-quantized (zp=0), so pass zp=0
    matmul_int8(gates_hh, gru->w_hh_int8, h_prev, gru->b_hh_int32,
                3 * H, H, gru->w_hh_scale, model->gru_out_scale, 0);
    
    for (int i = 0; i < H; i++) {
        r_hh[i] = gates_hh[i];
        z_hh[i] = gates_hh[H + i];
        n_hh[i] = gates_hh[2 * H + i];
    }
    
    // 3. Compute gates
    float r_t[HIDDEN_SIZE], z_t[HIDDEN_SIZE], n_t[HIDDEN_SIZE];
    
    for (int i = 0; i < H; i++) {
        // Reset gate: r_t = sigmoid(r_ih + r_hh)
        r_t[i] = sigmoid_f(r_ih[i] + r_hh[i]);
        
        // Update gate: z_t = sigmoid(z_ih + z_hh)
        z_t[i] = sigmoid_f(z_ih[i] + z_hh[i]);
        
        // New gate: n_t = tanh(n_ih + r_t * n_hh)
        n_t[i] = tanh_f(n_ih[i] + r_t[i] * n_hh[i]);
    }
    
    // 4. Compute new hidden state
    // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    for (int i = 0; i < H; i++) {
        float h_prev_f = dequantize_int8(h_prev[i], model->gru_out_scale, model->gru_out_zp);
        float h_new = (1.0f - z_t[i]) * n_t[i] + z_t[i] * h_prev_f;
        
        // Requantize
        h_out[i] = quantize_int8(h_new, model->gru_out_scale, model->gru_out_zp);
    }
}

/* ========================================
 *          Linear layer
 * ======================================== */

float int8_linear(const int8_t *input, const TinyGRUModel *model) {
    const LinearParams *head = &model->head;
    int H = head->in_features;
    
    int32_t acc = 0;
    
    for (int i = 0; i < H; i++) {
        acc += (int32_t)head->w_int8[i] * (int32_t)input[i];
    }
    
    // Add bias
    acc += head->b_int32[0];
    
    // Dequantize
    float scale = head->w_scale * model->gru_out_scale;
    return (float)acc * scale;
}

/* ========================================
 *          Full inference
 * ======================================== */

float inference_forward(const TinyGRUModel *model, 
                        InferenceBuffer *buf,
                        const uint8_t *input) {
    
    int I = model->input_size;
    int H = model->hidden_size;
    
    // 1. Reset hidden state
    inference_buffer_reset(buf, H);
    
    // 2. Quantize inputs and iterate timesteps
    int8_t x_t[INPUT_SIZE];
    
    for (int t = 0; t < WINDOW_SIZE; t++) {
        // Quantize input at current timestep
        for (int c = 0; c < I; c++) {
            float val = (float)input[t * I + c] / 255.0f;
            x_t[c] = quantize_int8(val, model->input_scale, model->input_zp);
        }
        
        // GRU cell computation
        int8_t h_new[HIDDEN_SIZE];
        gru_cell(h_new, buf->h_state, x_t, model, buf, 0);
        
        // Update hidden state
        memcpy(buf->h_state, h_new, H * sizeof(int8_t));
    }
    
    // 3. Head (Linear)
    return int8_linear(buf->h_state, model);
}

void inference_batch(const TinyGRUModel *model,
                     InferenceBuffer *buf,
                     const uint8_t *inputs,
                     float *outputs,
                     int batch_size) {
    
    int sample_size = WINDOW_SIZE * model->input_size;
    
    for (int i = 0; i < batch_size; i++) {
        outputs[i] = inference_forward(model, buf, inputs + i * sample_size);
    }
}

