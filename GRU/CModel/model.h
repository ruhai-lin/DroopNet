/**
 * @file model.h
 * @brief TinyGRU INT8 模型定义 - ASIC Golden Model
 * 
 * 模型结构:
 * - 单层 GRU, hidden_size = 36
 * - 输入: 9 通道, 50 时间步
 * - 输出: 1 (二分类 logit)
 * 
 * 内存占用估算 (INT8):
 * - GRU 权重: 3 * 36 * (9 + 36 + 2) = 5076 bytes
 * - Head 权重: 36 + 1 = 37 bytes
 * - 总权重: ~5.1 KB
 */

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>

/* ========================================
 *          模型架构常量
 * ======================================== */

#define INPUT_SIZE      9
#define HIDDEN_SIZE     36
#define NUM_LAYERS      1
#define WINDOW_SIZE     50

// GRU 门数 (reset, update, new)
#define NUM_GATES       3

/* ========================================
 *          量化参数结构
 * ======================================== */

/**
 * @brief GRU 层量化参数和权重
 */
typedef struct {
    // weight_ih: (3*H, I)
    float w_ih_scale;
    int32_t w_ih_zp;
    int8_t *w_ih_int8;      // [3*H * I]
    int32_t *b_ih_int32;    // [3*H]
    
    // weight_hh: (3*H, H)
    float w_hh_scale;
    int32_t w_hh_zp;
    int8_t *w_hh_int8;      // [3*H * H]
    int32_t *b_hh_int32;    // [3*H]
} GRULayerParams;

/**
 * @brief Linear 层量化参数和权重
 */
typedef struct {
    float w_scale;
    int32_t w_zp;
    int8_t *w_int8;         // [out_features * in_features]
    int32_t *b_int32;       // [out_features]
    
    int in_features;
    int out_features;
} LinearParams;

/**
 * @brief 完整 GRU 模型
 */
typedef struct {
    // 模型配置
    int input_size;
    int hidden_size;
    int num_layers;
    
    // 激活值量化参数
    float input_scale;
    int32_t input_zp;
    float gru_out_scale;
    int32_t gru_out_zp;
    float head_out_scale;
    int32_t head_out_zp;
    
    // 层参数
    GRULayerParams gru_layers[NUM_LAYERS];
    LinearParams head;
} TinyGRUModel;

/* ========================================
 *          模型加载/释放函数
 * ======================================== */

/**
 * @brief 从二进制文件加载模型权重
 * @param model 模型结构指针
 * @param filepath 权重文件路径
 * @return 0 成功, -1 失败
 */
int model_load(TinyGRUModel *model, const char *filepath);

/**
 * @brief 释放模型内存
 * @param model 模型结构指针
 */
void model_free(TinyGRUModel *model);

/**
 * @brief 打印模型信息
 * @param model 模型结构指针
 */
void model_print_info(const TinyGRUModel *model);

/**
 * @brief 计算模型权重大小 (字节)
 * @param model 模型结构指针
 * @return 权重字节数
 */
size_t model_weight_bytes(const TinyGRUModel *model);

#endif // MODEL_H

