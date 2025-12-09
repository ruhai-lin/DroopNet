/**
 * @file inference.c
 * @brief TinyTCN INT8 推理实现 - ASIC Golden Model (PTQ 版本)
 */

#include "inference.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================
 *          缓冲区管理
 * ======================================== */

int inference_buffer_init(InferenceBuffer *buf) {
    buf->buffer_size = MAX_CHANNELS * WINDOW_SIZE;
    
    buf->buffer_a = (int8_t *)calloc(buf->buffer_size, sizeof(int8_t));
    buf->buffer_b = (int8_t *)calloc(buf->buffer_size, sizeof(int8_t));
    buf->temp1 = (int8_t *)calloc(buf->buffer_size, sizeof(int8_t));
    buf->temp2 = (int8_t *)calloc(buf->buffer_size, sizeof(int8_t));
    
    if (!buf->buffer_a || !buf->buffer_b || !buf->temp1 || !buf->temp2) {
        inference_buffer_free(buf);
        return -1;
    }
    
    buf->current = buf->buffer_a;
    buf->next = buf->buffer_b;
    
    return 0;
}

void inference_buffer_free(InferenceBuffer *buf) {
    if (buf->buffer_a) { free(buf->buffer_a); buf->buffer_a = NULL; }
    if (buf->buffer_b) { free(buf->buffer_b); buf->buffer_b = NULL; }
    if (buf->temp1) { free(buf->temp1); buf->temp1 = NULL; }
    if (buf->temp2) { free(buf->temp2); buf->temp2 = NULL; }
}

static void swap_buffers(InferenceBuffer *buf) {
    int8_t *tmp = buf->current;
    buf->current = buf->next;
    buf->next = tmp;
}

/* ========================================
 *          量化基础运算
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
 *          INT8 Conv1D
 * ======================================== */

/**
 * @brief INT8 Conv1D 核心实现
 * 
 * 输入布局: [channels, seq_len] (CHW format)
 * 权重布局: [out_ch, in_ch, kernel_size]
 */
int int8_conv1d(int8_t *output,
                const int8_t *input,
                const LayerParams *params,
                int in_len,
                int dilation,
                int apply_relu) {
    
    int out_ch = params->out_channels;
    int in_ch = params->in_channels;
    int k = params->kernel_size;
    
    // Causal padding (左侧填充)
    int padding = (k - 1) * dilation;
    (void)padding;  // 用于计算，下面会用到
    
    // 输出长度 (因为有 chomp, 保持与输入相同)
    int out_len = in_len;
    
    int32_t x_zp = params->input_zp;
    
    // 遍历输出通道
    for (int oc = 0; oc < out_ch; oc++) {
        int32_t w_zp = params->weight_zps[oc];
        float w_scale = params->weight_scales[oc];
        float scale_factor = params->input_scale * w_scale;
        
        // 遍历输出时间步
        for (int t = 0; t < out_len; t++) {
            int32_t acc = 0;
            
            // 卷积核遍历
            for (int ic = 0; ic < in_ch; ic++) {
                for (int ki = 0; ki < k; ki++) {
                    // 计算输入索引 (考虑 padding 和 dilation)
                    int t_in = t + ki * dilation;  // 相对于 padded input
                    
                    int32_t x_val;
                    if (t_in < padding) {
                        // 在 padding 区域，使用 zero point
                        x_val = x_zp;
                    } else {
                        int actual_idx = t_in - padding;
                        if (actual_idx < in_len) {
                            x_val = (int32_t)input[ic * in_len + actual_idx];
                        } else {
                            x_val = x_zp;
                        }
                    }
                    
                    // 获取权重
                    int w_idx = oc * (in_ch * k) + ic * k + ki;
                    int32_t w_val = (int32_t)params->weights_int8[w_idx];
                    
                    // 累加: (x - x_zp) * (w - w_zp)
                    acc += (x_val - x_zp) * (w_val - w_zp);
                }
            }
            
            // 加上量化后的 bias
            acc += params->bias_int32[oc];
            
            // 反量化到 float
            float y_float = (float)acc * scale_factor;
            
            // ReLU (如果需要)
            if (apply_relu && y_float < 0) {
                y_float = 0.0f;
            }
            
            // 重新量化到输出 scale
            output[oc * out_len + t] = quantize_int8(y_float, 
                                                      params->output_scale, 
                                                      params->output_zp);
        }
    }
    
    return out_len;
}

/* ========================================
 *          INT8 残差相加
 * ======================================== */

void int8_add_relu(int8_t *output,
                   const int8_t *a, const int8_t *b,
                   int channels, int seq_len,
                   float a_scale, int32_t a_zp,
                   float b_scale, int32_t b_zp,
                   float out_scale, int32_t out_zp) {
    
    for (int c = 0; c < channels; c++) {
        for (int t = 0; t < seq_len; t++) {
            int idx = c * seq_len + t;
            
            // 反量化
            float a_float = dequantize_int8(a[idx], a_scale, a_zp);
            float b_float = dequantize_int8(b[idx], b_scale, b_zp);
            
            // 相加 + ReLU
            float sum = a_float + b_float;
            if (sum < 0) sum = 0.0f;
            
            // 重新量化
            output[idx] = quantize_int8(sum, out_scale, out_zp);
        }
    }
}

/* ========================================
 *          INT8 Linear
 * ======================================== */

float int8_linear(const int8_t *input, const LayerParams *params) {
    int in_features = params->in_channels;  // Linear: in_channels = in_features
    int32_t x_zp = params->input_zp;
    
    // 只有一个输出 (out_channels = 1)
    int32_t w_zp = params->weight_zps[0];
    float w_scale = params->weight_scales[0];
    
    int32_t acc = 0;
    
    for (int i = 0; i < in_features; i++) {
        int32_t x_val = (int32_t)input[i];
        int32_t w_val = (int32_t)params->weights_int8[i];
        
        acc += (x_val - x_zp) * (w_val - w_zp);
    }
    
    // 加 bias
    acc += params->bias_int32[0];
    
    // 反量化到 float (不需要重新量化，直接输出)
    float scale_factor = params->input_scale * w_scale;
    return (float)acc * scale_factor;
}

/* ========================================
 *          TCN Block 前向传播
 * ======================================== */

static void forward_block(const BlockParams *block,
                          int8_t *output,
                          const int8_t *input,
                          int8_t *temp_conv1,   // 用于 conv1 输出
                          int8_t *temp_conv2,   // 用于 conv2 输出
                          int out_ch,
                          int seq_len,
                          float *current_scale,
                          int32_t *current_zp) {
    
    // Conv1 + ReLU
    int8_conv1d(temp_conv1, input, &block->conv1, seq_len, 
                block->dilation, 1);  // apply_relu = 1
    
    // Conv2 + ReLU
    int8_conv1d(temp_conv2, temp_conv1, &block->conv2, seq_len,
                block->dilation, 1);  // apply_relu = 1
    
    // 残差分支
    const int8_t *residual;
    float res_scale;
    int32_t res_zp;
    
    if (block->has_downsample) {
        // Downsample (1x1 conv, 无 ReLU) - 使用 temp_conv1 作为输出 (conv1 已经用完)
        int8_conv1d(temp_conv1, input, &block->downsample, seq_len,
                    1, 0);  // dilation=1, no relu
        residual = temp_conv1;
        res_scale = block->downsample.output_scale;
        res_zp = block->downsample.output_zp;
    } else {
        residual = input;  // 直接使用输入
        res_scale = *current_scale;
        res_zp = *current_zp;
    }
    
    // 残差相加 + ReLU
    int8_add_relu(output,
                  residual, temp_conv2,
                  out_ch, seq_len,
                  res_scale, res_zp,
                  block->conv2.output_scale, block->conv2.output_zp,
                  block->block_out_scale, block->block_out_zp);
    
    // 更新当前量化参数
    *current_scale = block->block_out_scale;
    *current_zp = block->block_out_zp;
}

/* ========================================
 *          完整推理
 * ======================================== */

float inference_forward(const TinyTCNModel *model, 
                        InferenceBuffer *buf,
                        const uint8_t *input) {
    
    int seq_len = WINDOW_SIZE;
    
    // 1. 输入量化: uint8 [0-255] -> float [0-1] -> int8
    // 输入布局转换: [T, C] -> [C, T]
    for (int c = 0; c < NUM_INPUTS; c++) {
        for (int t = 0; t < seq_len; t++) {
            // 原始输入是 uint8 ADC code, 归一化到 [0, 1]
            float val = (float)input[t * NUM_INPUTS + c] / 255.0f;
            buf->current[c * seq_len + t] = quantize_int8(val, 
                                                          model->input_scale, 
                                                          model->input_zp);
        }
    }
    
    float current_scale = model->input_scale;
    int32_t current_zp = model->input_zp;
    
    // 输出通道数配置
    int out_channels[NUM_BLOCKS] = {CH_BLOCK0, CH_BLOCK1, CH_BLOCK2, CH_BLOCK3};
    
    // 2. 前向传播各 Block
    for (int i = 0; i < NUM_BLOCKS; i++) {
        forward_block(&model->blocks[i],
                      buf->next,        // output
                      buf->current,     // input
                      buf->temp1,       // temp for conv1
                      buf->temp2,       // temp for conv2
                      out_channels[i],
                      seq_len,
                      &current_scale, &current_zp);
        swap_buffers(buf);
    }
    
    // 3. 取最后一个时间步
    int8_t last_timestep[CH_BLOCK3];
    for (int c = 0; c < CH_BLOCK3; c++) {
        last_timestep[c] = buf->current[c * seq_len + (seq_len - 1)];
    }
    
    // 4. Head (Linear)
    return int8_linear(last_timestep, &model->head);
}

void inference_batch(const TinyTCNModel *model,
                     InferenceBuffer *buf,
                     const uint8_t *inputs,
                     float *outputs,
                     int batch_size) {
    
    int sample_size = WINDOW_SIZE * NUM_INPUTS;
    
    for (int i = 0; i < batch_size; i++) {
        outputs[i] = inference_forward(model, buf, inputs + i * sample_size);
    }
}

