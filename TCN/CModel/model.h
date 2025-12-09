/**
 * @file model.h
 * @brief TinyTCN INT8 模型定义 - ASIC Golden Model (PTQ 版本)
 * 
 * 模型结构:
 * - 4 个 TCN blocks, 通道数 [11, 10, 5, 4]
 * - kernel_size = 5, dilations = [1, 2, 4, 8]
 * - 输入: 9 通道, 50 时间步
 * - 输出: 1 (二分类 logit)
 */

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>

/* ========================================
 *          模型架构常量
 * ======================================== */

#define NUM_INPUTS      9
#define WINDOW_SIZE     50
#define KERNEL_SIZE     5
#define NUM_BLOCKS      4

// 各层通道数
#define CH_BLOCK0       11
#define CH_BLOCK1       10
#define CH_BLOCK2       5
#define CH_BLOCK3       4
#define CH_OUTPUT       1

// Dilations
#define DILATION_0      1
#define DILATION_1      2
#define DILATION_2      4
#define DILATION_3      8

// 最大通道数 (用于静态分配)
#define MAX_CHANNELS    11

/* ========================================
 *          量化层参数结构
 * ======================================== */

/**
 * @brief 单层量化参数
 */
typedef struct {
    float input_scale;
    int32_t input_zp;
    float output_scale;
    int32_t output_zp;
    
    int32_t out_channels;
    int32_t in_channels;
    int32_t kernel_size;
    
    float *weight_scales;      // [out_channels]
    int32_t *weight_zps;       // [out_channels]
    int32_t *bias_int32;       // [out_channels]
    int8_t *weights_int8;      // [out_channels, in_channels, kernel_size]
} LayerParams;

/**
 * @brief TCN Block 参数
 */
typedef struct {
    LayerParams conv1;
    LayerParams conv2;
    LayerParams downsample;    // 可能为空 (in_ch == out_ch 时)
    int has_downsample;
    
    float block_out_scale;
    int32_t block_out_zp;
    
    int dilation;
} BlockParams;

/**
 * @brief 完整模型参数
 */
typedef struct {
    BlockParams blocks[NUM_BLOCKS];
    LayerParams head;           // Linear layer
    
    // 初始量化参数 (从第一层获取)
    float input_scale;
    int32_t input_zp;
} TinyTCNModel;

/* ========================================
 *          模型加载/释放函数
 * ======================================== */

/**
 * @brief 从二进制文件加载模型权重
 * @param model 模型结构指针
 * @param filepath 权重文件路径
 * @return 0 成功, -1 失败
 */
int model_load(TinyTCNModel *model, const char *filepath);

/**
 * @brief 释放模型内存
 * @param model 模型结构指针
 */
void model_free(TinyTCNModel *model);

/**
 * @brief 打印模型信息
 * @param model 模型结构指针
 */
void model_print_info(const TinyTCNModel *model);

/**
 * @brief 计算模型权重大小 (字节)
 * @param model 模型结构指针
 * @return 权重字节数
 */
size_t model_weight_bytes(const TinyTCNModel *model);

#endif // MODEL_H

