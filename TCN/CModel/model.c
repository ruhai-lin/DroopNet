/**
 * @file model.c
 * @brief TinyTCN INT8 model loading implementation (PTQ version)
 */

#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================
 *          Internal helpers
 * ======================================== */

static int read_f32(FILE *f, float *val) {
    return fread(val, sizeof(float), 1, f) == 1 ? 0 : -1;
}

static int read_i32(FILE *f, int32_t *val) {
    return fread(val, sizeof(int32_t), 1, f) == 1 ? 0 : -1;
}

static int read_f32_array(FILE *f, float *arr, int count) {
    return fread(arr, sizeof(float), count, f) == (size_t)count ? 0 : -1;
}

static int read_i32_array(FILE *f, int32_t *arr, int count) {
    return fread(arr, sizeof(int32_t), count, f) == (size_t)count ? 0 : -1;
}

static int read_i8_array(FILE *f, int8_t *arr, int count) {
    return fread(arr, sizeof(int8_t), count, f) == (size_t)count ? 0 : -1;
}

/**
 * @brief Read parameters for a single layer
 */
static int load_layer_params(FILE *f, LayerParams *layer, 
                             int out_ch, int in_ch, int kernel) {
    layer->out_channels = out_ch;
    layer->in_channels = in_ch;
    layer->kernel_size = kernel;
    
    // Read quantization parameters
    if (read_f32(f, &layer->input_scale) < 0) return -1;
    if (read_i32(f, &layer->input_zp) < 0) return -1;
    if (read_f32(f, &layer->output_scale) < 0) return -1;
    if (read_i32(f, &layer->output_zp) < 0) return -1;
    
    // Allocate and read weight parameters
    layer->weight_scales = (float *)malloc(out_ch * sizeof(float));
    layer->weight_zps = (int32_t *)malloc(out_ch * sizeof(int32_t));
    layer->bias_int32 = (int32_t *)malloc(out_ch * sizeof(int32_t));
    
    int weight_count = out_ch * in_ch * kernel;
    layer->weights_int8 = (int8_t *)malloc(weight_count * sizeof(int8_t));
    
    if (!layer->weight_scales || !layer->weight_zps || 
        !layer->bias_int32 || !layer->weights_int8) {
        return -1;
    }
    
    if (read_f32_array(f, layer->weight_scales, out_ch) < 0) return -1;
    if (read_i32_array(f, layer->weight_zps, out_ch) < 0) return -1;
    if (read_i32_array(f, layer->bias_int32, out_ch) < 0) return -1;
    if (read_i8_array(f, layer->weights_int8, weight_count) < 0) return -1;
    
    return 0;
}

/**
 * @brief Free memory for a single layer
 */
static void free_layer_params(LayerParams *layer) {
    if (layer->weight_scales) { free(layer->weight_scales); layer->weight_scales = NULL; }
    if (layer->weight_zps) { free(layer->weight_zps); layer->weight_zps = NULL; }
    if (layer->bias_int32) { free(layer->bias_int32); layer->bias_int32 = NULL; }
    if (layer->weights_int8) { free(layer->weights_int8); layer->weights_int8 = NULL; }
}

/* ========================================
 *          Public API
 * ======================================== */

int model_load(TinyTCNModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[Error] Cannot open weight file: %s\n", filepath);
        return -1;
    }
    
    memset(model, 0, sizeof(TinyTCNModel));
    
    // Block configuration
    int in_channels[NUM_BLOCKS] = {NUM_INPUTS, CH_BLOCK0, CH_BLOCK1, CH_BLOCK2};
    int out_channels[NUM_BLOCKS] = {CH_BLOCK0, CH_BLOCK1, CH_BLOCK2, CH_BLOCK3};
    int dilations[NUM_BLOCKS] = {DILATION_0, DILATION_1, DILATION_2, DILATION_3};
    
    // Load each block
    for (int i = 0; i < NUM_BLOCKS; i++) {
        BlockParams *block = &model->blocks[i];
        int in_ch = in_channels[i];
        int out_ch = out_channels[i];
        
        block->dilation = dilations[i];
        
        // Conv1: in_ch -> out_ch
        if (load_layer_params(f, &block->conv1, out_ch, in_ch, KERNEL_SIZE) < 0) {
            fprintf(stderr, "[Error] Failed to load block %d conv1\n", i);
            goto error;
        }
        
        // Save initial input quantization parameters
        if (i == 0) {
            model->input_scale = block->conv1.input_scale;
            model->input_zp = block->conv1.input_zp;
        }
        
        // Conv2: out_ch -> out_ch
        if (load_layer_params(f, &block->conv2, out_ch, out_ch, KERNEL_SIZE) < 0) {
            fprintf(stderr, "[Error] Failed to load block %d conv2\n", i);
            goto error;
        }
        
        // Downsample (1x1 conv) - only when in_ch != out_ch
        if (in_ch != out_ch) {
            block->has_downsample = 1;
            if (load_layer_params(f, &block->downsample, out_ch, in_ch, 1) < 0) {
                fprintf(stderr, "[Error] Failed to load block %d downsample\n", i);
                goto error;
            }
        } else {
            block->has_downsample = 0;
        }
        
        // Block output quantization parameters
        if (read_f32(f, &block->block_out_scale) < 0) goto error;
        if (read_i32(f, &block->block_out_zp) < 0) goto error;
    }
    
    // Head (Linear): CH_BLOCK3 -> 1
    if (load_layer_params(f, &model->head, CH_OUTPUT, CH_BLOCK3, 1) < 0) {
        fprintf(stderr, "[Error] Failed to load head layer\n");
        goto error;
    }
    
    fclose(f);
    printf("[Model] Loaded successfully from %s\n", filepath);
    return 0;

error:
    fclose(f);
    model_free(model);
    return -1;
}

void model_free(TinyTCNModel *model) {
    for (int i = 0; i < NUM_BLOCKS; i++) {
        free_layer_params(&model->blocks[i].conv1);
        free_layer_params(&model->blocks[i].conv2);
        if (model->blocks[i].has_downsample) {
            free_layer_params(&model->blocks[i].downsample);
        }
    }
    free_layer_params(&model->head);
}

void model_print_info(const TinyTCNModel *model) {
    printf("\n======== TinyTCN INT8 Model Info (PTQ) ========\n");
    printf("Input: %d channels x %d timesteps\n", NUM_INPUTS, WINDOW_SIZE);
    printf("Blocks: %d\n", NUM_BLOCKS);
    
    int in_channels[NUM_BLOCKS] = {NUM_INPUTS, CH_BLOCK0, CH_BLOCK1, CH_BLOCK2};
    int out_channels[NUM_BLOCKS] = {CH_BLOCK0, CH_BLOCK1, CH_BLOCK2, CH_BLOCK3};
    
    for (int i = 0; i < NUM_BLOCKS; i++) {
        const BlockParams *b = &model->blocks[i];
        printf("  Block %d: %d -> %d, dilation=%d, downsample=%s\n",
               i, in_channels[i], out_channels[i], b->dilation,
               b->has_downsample ? "yes" : "no");
    }
    
    printf("Head: %d -> %d (Linear)\n", CH_BLOCK3, CH_OUTPUT);
    printf("Initial Quant: scale=%.6f, zp=%d\n", 
           model->input_scale, model->input_zp);
    printf("Weight Size: %zu bytes\n", model_weight_bytes(model));
    printf("================================================\n\n");
}

size_t model_weight_bytes(const TinyTCNModel *model) {
    size_t total = 0;
    
    for (int i = 0; i < NUM_BLOCKS; i++) {
        const BlockParams *b = &model->blocks[i];
        
        // Conv1 weights
        total += b->conv1.out_channels * b->conv1.in_channels * b->conv1.kernel_size;
        // Conv2 weights
        total += b->conv2.out_channels * b->conv2.in_channels * b->conv2.kernel_size;
        // Downsample weights
        if (b->has_downsample) {
            total += b->downsample.out_channels * b->downsample.in_channels * b->downsample.kernel_size;
        }
    }
    
    // Head weights
    total += model->head.out_channels * model->head.in_channels * model->head.kernel_size;
    
    return total;
}

