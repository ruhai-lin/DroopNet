/**
 * @file inference.h
 * @brief TinyTCN INT8 inference function interface (PTQ version)
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "model.h"
#include <stdint.h>

/* ========================================
 *          Inference buffers
 * ======================================== */

/**
 * @brief Inference working buffer
 * Used to store intermediate activations and avoid dynamic allocation
 * 
 * Four-buffer design:
 * - buffer_a/b: alternate between blocks for input/output
 * - temp1/temp2: conv1/conv2 outputs within a block
 */
typedef struct {
    int8_t *buffer_a;   // [MAX_CHANNELS * WINDOW_SIZE]
    int8_t *buffer_b;   // [MAX_CHANNELS * WINDOW_SIZE]
    int8_t *temp1;      // [MAX_CHANNELS * WINDOW_SIZE] conv1 output
    int8_t *temp2;      // [MAX_CHANNELS * WINDOW_SIZE] conv2 output
    
    // Current buffer pointer (points to buffer_a or buffer_b)
    int8_t *current;
    int8_t *next;
    
    // Single buffer size
    size_t buffer_size;
} InferenceBuffer;

/**
 * @brief Initialize inference buffer
 * @param buf buffer struct pointer
 * @return 0 success, -1 failure
 */
int inference_buffer_init(InferenceBuffer *buf);

/**
 * @brief Free inference buffer
 * @param buf buffer struct pointer
 */
void inference_buffer_free(InferenceBuffer *buf);

/* ========================================
 *          Inference functions
 * ======================================== */

/**
 * @brief Run inference for a single sample
 * 
 * @param model model parameters
 * @param buf working buffer
 * @param input input data [WINDOW_SIZE * NUM_INPUTS], uint8
 * @return output logit (float)
 */
float inference_forward(const TinyTCNModel *model, 
                        InferenceBuffer *buf,
                        const uint8_t *input);

/**
 * @brief Batch inference
 * 
 * @param model model parameters
 * @param buf working buffer
 * @param inputs input data array [batch_size][WINDOW_SIZE * NUM_INPUTS]
 * @param outputs output array [batch_size]
 * @param batch_size batch size
 */
void inference_batch(const TinyTCNModel *model,
                     InferenceBuffer *buf,
                     const uint8_t *inputs,
                     float *outputs,
                     int batch_size);

/* ========================================
 *          Low-level quantized ops (ASIC reference)
 * ======================================== */

/**
 * @brief INT8 quantization
 * @param val float value
 * @param scale quantization scale
 * @param zp quantization zero point
 * @return quantized int8 value
 */
int8_t quantize_int8(float val, float scale, int32_t zp);

/**
 * @brief INT8 dequantization
 * @param val int8 value
 * @param scale quantization scale
 * @param zp quantization zero point
 * @return dequantized float value
 */
float dequantize_int8(int8_t val, float scale, int32_t zp);

/**
 * @brief INT8 Conv1D + ReLU
 * 
 * Quantized convolution formula:
 * y_int32 = sum((x - x_zp) * (w - w_zp)) + bias_int32
 * y_float = y_int32 * (input_scale * weight_scale)
 * y_int8 = quantize(relu(y_float))
 * 
 * @param output output buffer [out_ch * out_len]
 * @param input input data [in_ch * in_len]
 * @param params layer parameters
 * @param in_len input sequence length
 * @param dilation dilation factor
 * @param apply_relu whether to apply ReLU
 * @return output sequence length
 */
int int8_conv1d(int8_t *output,
                const int8_t *input,
                const LayerParams *params,
                int in_len,
                int dilation,
                int apply_relu);

/**
 * @brief INT8 residual add + ReLU
 * 
 * @param output output buffer
 * @param a input a
 * @param b input b
 * @param len length
 * @param a_scale scale for a
 * @param a_zp zero point for a
 * @param b_scale scale for b
 * @param b_zp zero point for b
 * @param out_scale output scale
 * @param out_zp output zero point
 */
void int8_add_relu(int8_t *output,
                   const int8_t *a, const int8_t *b,
                   int channels, int seq_len,
                   float a_scale, int32_t a_zp,
                   float b_scale, int32_t b_zp,
                   float out_scale, int32_t out_zp);

/**
 * @brief INT8 Linear layer (float output)
 * 
 * @param input input data [in_features]
 * @param params layer parameters
 * @return output value (float)
 */
float int8_linear(const int8_t *input, const LayerParams *params);

#endif // INFERENCE_H

