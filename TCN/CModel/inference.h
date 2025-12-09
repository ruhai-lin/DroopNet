/**
 * @file inference.h
 * @brief TinyTCN INT8 推理函数接口 (PTQ 版本)
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "model.h"
#include <stdint.h>

/* ========================================
 *          推理缓冲区
 * ======================================== */

/**
 * @brief 推理工作缓冲区
 * 用于存储中间激活值，避免动态分配
 * 
 * 四缓冲设计:
 * - buffer_a/b: 用于 block 间的 input/output 交替
 * - temp1/temp2: 用于 block 内部的 conv1/conv2 输出
 */
typedef struct {
    int8_t *buffer_a;   // [MAX_CHANNELS * WINDOW_SIZE]
    int8_t *buffer_b;   // [MAX_CHANNELS * WINDOW_SIZE]
    int8_t *temp1;      // [MAX_CHANNELS * WINDOW_SIZE] conv1 输出
    int8_t *temp2;      // [MAX_CHANNELS * WINDOW_SIZE] conv2 输出
    
    // 当前缓冲区指针 (指向 buffer_a 或 buffer_b)
    int8_t *current;
    int8_t *next;
    
    // 单缓冲区大小
    size_t buffer_size;
} InferenceBuffer;

/**
 * @brief 初始化推理缓冲区
 * @param buf 缓冲区结构指针
 * @return 0 成功, -1 失败
 */
int inference_buffer_init(InferenceBuffer *buf);

/**
 * @brief 释放推理缓冲区
 * @param buf 缓冲区结构指针
 */
void inference_buffer_free(InferenceBuffer *buf);

/* ========================================
 *          推理函数
 * ======================================== */

/**
 * @brief 执行单样本推理
 * 
 * @param model 模型参数
 * @param buf 工作缓冲区
 * @param input 输入数据 [WINDOW_SIZE * NUM_INPUTS], uint8
 * @return 输出 logit (float)
 */
float inference_forward(const TinyTCNModel *model, 
                        InferenceBuffer *buf,
                        const uint8_t *input);

/**
 * @brief 批量推理
 * 
 * @param model 模型参数
 * @param buf 工作缓冲区
 * @param inputs 输入数据数组 [batch_size][WINDOW_SIZE * NUM_INPUTS]
 * @param outputs 输出数组 [batch_size]
 * @param batch_size 批次大小
 */
void inference_batch(const TinyTCNModel *model,
                     InferenceBuffer *buf,
                     const uint8_t *inputs,
                     float *outputs,
                     int batch_size);

/* ========================================
 *          底层量化运算 (可用于 ASIC 参考)
 * ======================================== */

/**
 * @brief INT8 量化
 * @param val 浮点值
 * @param scale 量化 scale
 * @param zp 量化 zero point
 * @return 量化后的 int8 值
 */
int8_t quantize_int8(float val, float scale, int32_t zp);

/**
 * @brief INT8 反量化
 * @param val int8 值
 * @param scale 量化 scale
 * @param zp 量化 zero point
 * @return 反量化后的浮点值
 */
float dequantize_int8(int8_t val, float scale, int32_t zp);

/**
 * @brief INT8 Conv1D + ReLU
 * 
 * 量化卷积公式:
 * y_int32 = sum((x - x_zp) * (w - w_zp)) + bias_int32
 * y_float = y_int32 * (input_scale * weight_scale)
 * y_int8 = quantize(relu(y_float))
 * 
 * @param output 输出缓冲区 [out_ch * out_len]
 * @param input 输入数据 [in_ch * in_len]
 * @param params 层参数
 * @param in_len 输入序列长度
 * @param dilation 膨胀率
 * @param apply_relu 是否应用 ReLU
 * @return 输出序列长度
 */
int int8_conv1d(int8_t *output,
                const int8_t *input,
                const LayerParams *params,
                int in_len,
                int dilation,
                int apply_relu);

/**
 * @brief INT8 残差相加 + ReLU
 * 
 * @param output 输出缓冲区
 * @param a 输入 a
 * @param b 输入 b
 * @param len 长度
 * @param a_scale a 的 scale
 * @param a_zp a 的 zero point
 * @param b_scale b 的 scale
 * @param b_zp b 的 zero point
 * @param out_scale 输出 scale
 * @param out_zp 输出 zero point
 */
void int8_add_relu(int8_t *output,
                   const int8_t *a, const int8_t *b,
                   int channels, int seq_len,
                   float a_scale, int32_t a_zp,
                   float b_scale, int32_t b_zp,
                   float out_scale, int32_t out_zp);

/**
 * @brief INT8 Linear 层 (输出为 float)
 * 
 * @param input 输入数据 [in_features]
 * @param params 层参数
 * @return 输出值 (float)
 */
float int8_linear(const int8_t *input, const LayerParams *params);

#endif // INFERENCE_H

