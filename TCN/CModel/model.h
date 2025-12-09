/**
 * @file model.h
 * @brief TinyTCN INT8 model definition - ASIC Golden Model (PTQ version)
 * 
 * Model structure:
 * - 4 TCN blocks, channels [11, 10, 5, 4]
 * - kernel_size = 5, dilations = [1, 2, 4, 8]
 * - Input: 9 channels, 50 timesteps
 * - Output: 1 (binary classification logit)
 */

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>

/* ========================================
 *          Model architecture constants
 * ======================================== */

#define NUM_INPUTS      9
#define WINDOW_SIZE     50
#define KERNEL_SIZE     5
#define NUM_BLOCKS      4

// Channels per block
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

// Max channels (for static allocation)
#define MAX_CHANNELS    11

/* ========================================
 *          Quantized layer parameter structs
 * ======================================== */

/**
 * @brief Quantization parameters for a single layer
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
 * @brief TCN block parameters
 */
typedef struct {
    LayerParams conv1;
    LayerParams conv2;
    LayerParams downsample;    // may be empty (when in_ch == out_ch)
    int has_downsample;
    
    float block_out_scale;
    int32_t block_out_zp;
    
    int dilation;
} BlockParams;

/**
 * @brief Full model parameters
 */
typedef struct {
    BlockParams blocks[NUM_BLOCKS];
    LayerParams head;           // Linear layer
    
    // Initial quantization params (from first layer)
    float input_scale;
    int32_t input_zp;
} TinyTCNModel;

/* ========================================
 *          Model load/free helpers
 * ======================================== */

/**
 * @brief Load model weights from binary file
 * @param model model struct pointer
 * @param filepath weight file path
 * @return 0 success, -1 failure
 */
int model_load(TinyTCNModel *model, const char *filepath);

/**
 * @brief Free model memory
 * @param model model struct pointer
 */
void model_free(TinyTCNModel *model);

/**
 * @brief Print model info
 * @param model model struct pointer
 */
void model_print_info(const TinyTCNModel *model);

/**
 * @brief Calculate model weight size (bytes)
 * @param model model struct pointer
 * @return weight bytes
 */
size_t model_weight_bytes(const TinyTCNModel *model);

#endif // MODEL_H

