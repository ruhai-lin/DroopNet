/**
 * @file inference.h
 * @brief TinyGRU INT8 inference function interface
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
 * Used to store hidden state and intermediate results
 */
typedef struct {
    int8_t *h_state;        // current hidden state [HIDDEN_SIZE]
    int32_t *acc_buffer;    // INT32 accumulation buffer [3 * HIDDEN_SIZE]
    float *gate_buffer;     // gate computation float buffer [3 * HIDDEN_SIZE]
} InferenceBuffer;

/**
 * @brief Initialize inference buffer
 * @param buf buffer struct pointer
 * @param hidden_size hidden layer size
 * @return 0 success, -1 failure
 */
int inference_buffer_init(InferenceBuffer *buf, int hidden_size);

/**
 * @brief Free inference buffer
 * @param buf buffer struct pointer
 */
void inference_buffer_free(InferenceBuffer *buf);

/**
 * @brief Reset hidden state to zero
 * @param buf buffer struct pointer
 * @param hidden_size hidden layer size
 */
void inference_buffer_reset(InferenceBuffer *buf, int hidden_size);

/* ========================================
 *          Inference functions
 * ======================================== */

/**
 * @brief Run inference for a single sample
 * 
 * @param model model parameters
 * @param buf working buffer
 * @param input input data [WINDOW_SIZE * INPUT_SIZE], uint8
 * @return output logit (float)
 */
float inference_forward(const TinyGRUModel *model, 
                        InferenceBuffer *buf,
                        const uint8_t *input);

/**
 * @brief Batch inference
 * 
 * @param model model parameters
 * @param buf working buffer
 * @param inputs input data array [batch_size][WINDOW_SIZE * INPUT_SIZE]
 * @param outputs output array [batch_size]
 * @param batch_size batch size
 */
void inference_batch(const TinyGRUModel *model,
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
 * @brief GRU cell single step
 * 
 * @param h_out output hidden state [hidden_size], int8
 * @param h_prev previous hidden state [hidden_size], int8
 * @param x_t current input [input_size], int8
 * @param model model parameters
 * @param buf working buffer
 * @param layer_idx layer index
 */
void gru_cell(int8_t *h_out,
              const int8_t *h_prev,
              const int8_t *x_t,
              const TinyGRUModel *model,
              InferenceBuffer *buf,
              int layer_idx);

/**
 * @brief INT8 Linear layer (float output)
 * 
 * @param input input data [in_features], int8
 * @param model model parameters
 * @return output value (float)
 */
float int8_linear(const int8_t *input, const TinyGRUModel *model);

#endif // INFERENCE_H

