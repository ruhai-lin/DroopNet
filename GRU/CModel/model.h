/**
 * @file model.h
 * @brief TinyGRU INT8 model definition - ASIC Golden Model
 * 
 * Model structure:
 * - Single-layer GRU, hidden_size = 36
 * - Input: 9 channels, 50 timesteps
 * - Output: 1 (binary classification logit)
 * 
 * Memory estimate (INT8):
 * - GRU weights: 3 * 36 * (9 + 36 + 2) = 5076 bytes
 * - Head weights: 36 + 1 = 37 bytes
 * - Total weights: ~5.1 KB
 */

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>

/* ========================================
 *          Model architecture constants
 * ======================================== */

#define INPUT_SIZE      9
#define HIDDEN_SIZE     36
#define NUM_LAYERS      1
#define WINDOW_SIZE     50

// Number of GRU gates (reset, update, new)
#define NUM_GATES       3

/* ========================================
 *          Quantization parameter structs
 * ======================================== */

/**
 * @brief GRU layer quantization parameters and weights
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
 * @brief Linear layer quantization parameters and weights
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
 * @brief Complete GRU model
 */
typedef struct {
    // Model configuration
    int input_size;
    int hidden_size;
    int num_layers;
    
    // Activation quantization parameters
    float input_scale;
    int32_t input_zp;
    float gru_out_scale;
    int32_t gru_out_zp;
    float head_out_scale;
    int32_t head_out_zp;
    
    // Layer parameters
    GRULayerParams gru_layers[NUM_LAYERS];
    LinearParams head;
} TinyGRUModel;

/* ========================================
 *          Model load/free helpers
 * ======================================== */

/**
 * @brief Load model weights from binary file
 * @param model model struct pointer
 * @param filepath weight file path
 * @return 0 success, -1 failure
 */
int model_load(TinyGRUModel *model, const char *filepath);

/**
 * @brief Free model memory
 * @param model model struct pointer
 */
void model_free(TinyGRUModel *model);

/**
 * @brief Print model info
 * @param model model struct pointer
 */
void model_print_info(const TinyGRUModel *model);

/**
 * @brief Calculate model weight size (bytes)
 * @param model model struct pointer
 * @return weight bytes
 */
size_t model_weight_bytes(const TinyGRUModel *model);

#endif // MODEL_H

