/**
 * @file inference.h
 * @brief TinyGRU INT8 推理函数接口
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
 * 用于存储隐藏状态和中间计算结果
 */
typedef struct {
    int8_t *h_state;        // 当前隐藏状态 [HIDDEN_SIZE]
    int32_t *acc_buffer;    // INT32 累加缓冲区 [3 * HIDDEN_SIZE]
    float *gate_buffer;     // 门计算浮点缓冲区 [3 * HIDDEN_SIZE]
} InferenceBuffer;

/**
 * @brief 初始化推理缓冲区
 * @param buf 缓冲区结构指针
 * @param hidden_size 隐藏层大小
 * @return 0 成功, -1 失败
 */
int inference_buffer_init(InferenceBuffer *buf, int hidden_size);

/**
 * @brief 释放推理缓冲区
 * @param buf 缓冲区结构指针
 */
void inference_buffer_free(InferenceBuffer *buf);

/**
 * @brief 重置隐藏状态为零
 * @param buf 缓冲区结构指针
 * @param hidden_size 隐藏层大小
 */
void inference_buffer_reset(InferenceBuffer *buf, int hidden_size);

/* ========================================
 *          推理函数
 * ======================================== */

/**
 * @brief 执行单样本推理
 * 
 * @param model 模型参数
 * @param buf 工作缓冲区
 * @param input 输入数据 [WINDOW_SIZE * INPUT_SIZE], uint8
 * @return 输出 logit (float)
 */
float inference_forward(const TinyGRUModel *model, 
                        InferenceBuffer *buf,
                        const uint8_t *input);

/**
 * @brief 批量推理
 * 
 * @param model 模型参数
 * @param buf 工作缓冲区
 * @param inputs 输入数据数组 [batch_size][WINDOW_SIZE * INPUT_SIZE]
 * @param outputs 输出数组 [batch_size]
 * @param batch_size 批次大小
 */
void inference_batch(const TinyGRUModel *model,
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
 * @brief GRU Cell 单步计算
 * 
 * @param h_out 输出隐藏状态 [hidden_size], int8
 * @param h_prev 前一隐藏状态 [hidden_size], int8
 * @param x_t 当前输入 [input_size], int8
 * @param model 模型参数
 * @param buf 工作缓冲区
 * @param layer_idx 层索引
 */
void gru_cell(int8_t *h_out,
              const int8_t *h_prev,
              const int8_t *x_t,
              const TinyGRUModel *model,
              InferenceBuffer *buf,
              int layer_idx);

/**
 * @brief INT8 Linear 层 (输出为 float)
 * 
 * @param input 输入数据 [in_features], int8
 * @param model 模型参数
 * @return 输出值 (float)
 */
float int8_linear(const int8_t *input, const TinyGRUModel *model);

#endif // INFERENCE_H

