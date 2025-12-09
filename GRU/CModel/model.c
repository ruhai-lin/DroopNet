/**
 * @file model.c
 * @brief TinyGRU INT8 模型加载实现
 */

#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================
 *          内部辅助函数
 * ======================================== */

static int read_u32(FILE *f, uint32_t *val) {
    return fread(val, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int read_f32(FILE *f, float *val) {
    return fread(val, sizeof(float), 1, f) == 1 ? 0 : -1;
}

static int read_i32(FILE *f, int32_t *val) {
    return fread(val, sizeof(int32_t), 1, f) == 1 ? 0 : -1;
}

static int read_i8_array(FILE *f, int8_t *arr, int count) {
    return fread(arr, sizeof(int8_t), count, f) == (size_t)count ? 0 : -1;
}

static int read_i32_array(FILE *f, int32_t *arr, int count) {
    return fread(arr, sizeof(int32_t), count, f) == (size_t)count ? 0 : -1;
}

/* ========================================
 *          公共 API 实现
 * ======================================== */

int model_load(TinyGRUModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[Error] Cannot open weight file: %s\n", filepath);
        return -1;
    }
    
    memset(model, 0, sizeof(TinyGRUModel));
    
    // 读取 Header
    uint32_t magic;
    if (read_u32(f, &magic) < 0 || magic != 0x47525538) {
        fprintf(stderr, "[Error] Invalid magic number: 0x%08X\n", magic);
        fclose(f);
        return -1;
    }
    
    int32_t input_size, hidden_size, num_layers;
    if (read_i32(f, &input_size) < 0 ||
        read_i32(f, &hidden_size) < 0 ||
        read_i32(f, &num_layers) < 0) {
        fprintf(stderr, "[Error] Failed to read header\n");
        fclose(f);
        return -1;
    }
    
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->num_layers = num_layers;
    
    int H = hidden_size;
    int I = input_size;
    
    // 读取激活值量化参数
    if (read_f32(f, &model->input_scale) < 0 ||
        read_i32(f, &model->input_zp) < 0 ||
        read_f32(f, &model->gru_out_scale) < 0 ||
        read_i32(f, &model->gru_out_zp) < 0 ||
        read_f32(f, &model->head_out_scale) < 0 ||
        read_i32(f, &model->head_out_zp) < 0) {
        fprintf(stderr, "[Error] Failed to read activation scales\n");
        fclose(f);
        return -1;
    }
    
    // 读取 GRU 层
    for (int layer = 0; layer < num_layers; layer++) {
        GRULayerParams *gru = &model->gru_layers[layer];
        
        // weight_ih
        if (read_f32(f, &gru->w_ih_scale) < 0 ||
            read_i32(f, &gru->w_ih_zp) < 0) {
            fprintf(stderr, "[Error] Failed to read w_ih params\n");
            goto error;
        }
        
        int w_ih_size = 3 * H * I;
        gru->w_ih_int8 = (int8_t *)malloc(w_ih_size);
        if (!gru->w_ih_int8 || read_i8_array(f, gru->w_ih_int8, w_ih_size) < 0) {
            fprintf(stderr, "[Error] Failed to read w_ih weights\n");
            goto error;
        }
        
        // weight_hh
        if (read_f32(f, &gru->w_hh_scale) < 0 ||
            read_i32(f, &gru->w_hh_zp) < 0) {
            fprintf(stderr, "[Error] Failed to read w_hh params\n");
            goto error;
        }
        
        int w_hh_size = 3 * H * H;
        gru->w_hh_int8 = (int8_t *)malloc(w_hh_size);
        if (!gru->w_hh_int8 || read_i8_array(f, gru->w_hh_int8, w_hh_size) < 0) {
            fprintf(stderr, "[Error] Failed to read w_hh weights\n");
            goto error;
        }
        
        // bias_ih
        gru->b_ih_int32 = (int32_t *)malloc(3 * H * sizeof(int32_t));
        if (!gru->b_ih_int32 || read_i32_array(f, gru->b_ih_int32, 3 * H) < 0) {
            fprintf(stderr, "[Error] Failed to read b_ih\n");
            goto error;
        }
        
        // bias_hh
        gru->b_hh_int32 = (int32_t *)malloc(3 * H * sizeof(int32_t));
        if (!gru->b_hh_int32 || read_i32_array(f, gru->b_hh_int32, 3 * H) < 0) {
            fprintf(stderr, "[Error] Failed to read b_hh\n");
            goto error;
        }
    }
    
    // 读取 Head 层
    LinearParams *head = &model->head;
    head->in_features = H;
    head->out_features = 1;
    
    if (read_f32(f, &head->w_scale) < 0 ||
        read_i32(f, &head->w_zp) < 0) {
        fprintf(stderr, "[Error] Failed to read head params\n");
        goto error;
    }
    
    head->w_int8 = (int8_t *)malloc(H);
    if (!head->w_int8 || read_i8_array(f, head->w_int8, H) < 0) {
        fprintf(stderr, "[Error] Failed to read head weights\n");
        goto error;
    }
    
    head->b_int32 = (int32_t *)malloc(sizeof(int32_t));
    if (!head->b_int32 || read_i32_array(f, head->b_int32, 1) < 0) {
        fprintf(stderr, "[Error] Failed to read head bias\n");
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

void model_free(TinyGRUModel *model) {
    for (int i = 0; i < model->num_layers; i++) {
        GRULayerParams *gru = &model->gru_layers[i];
        if (gru->w_ih_int8) { free(gru->w_ih_int8); gru->w_ih_int8 = NULL; }
        if (gru->w_hh_int8) { free(gru->w_hh_int8); gru->w_hh_int8 = NULL; }
        if (gru->b_ih_int32) { free(gru->b_ih_int32); gru->b_ih_int32 = NULL; }
        if (gru->b_hh_int32) { free(gru->b_hh_int32); gru->b_hh_int32 = NULL; }
    }
    
    if (model->head.w_int8) { free(model->head.w_int8); model->head.w_int8 = NULL; }
    if (model->head.b_int32) { free(model->head.b_int32); model->head.b_int32 = NULL; }
}

void model_print_info(const TinyGRUModel *model) {
    printf("\n======== TinyGRU INT8 Model Info ========\n");
    printf("Input: %d channels x %d timesteps\n", model->input_size, WINDOW_SIZE);
    printf("Hidden Size: %d\n", model->hidden_size);
    printf("Num Layers: %d\n", model->num_layers);
    printf("\nQuantization Params:\n");
    printf("  Input:    scale=%.6f, zp=%d\n", model->input_scale, model->input_zp);
    printf("  GRU Out:  scale=%.6f, zp=%d\n", model->gru_out_scale, model->gru_out_zp);
    printf("  Head Out: scale=%.6f, zp=%d\n", model->head_out_scale, model->head_out_zp);
    printf("\nGRU Layer 0:\n");
    printf("  w_ih: scale=%.6f, shape=(%d, %d)\n", 
           model->gru_layers[0].w_ih_scale, 3 * model->hidden_size, model->input_size);
    printf("  w_hh: scale=%.6f, shape=(%d, %d)\n",
           model->gru_layers[0].w_hh_scale, 3 * model->hidden_size, model->hidden_size);
    printf("\nHead: %d -> 1 (Linear)\n", model->hidden_size);
    printf("Weight Size: %zu bytes\n", model_weight_bytes(model));
    printf("==========================================\n\n");
}

size_t model_weight_bytes(const TinyGRUModel *model) {
    int H = model->hidden_size;
    int I = model->input_size;
    
    size_t total = 0;
    
    // GRU 层
    for (int i = 0; i < model->num_layers; i++) {
        total += 3 * H * I;     // w_ih
        total += 3 * H * H;     // w_hh
        // bias 以 int32 存储，但这里只计算 int8 等效
    }
    
    // Head 层
    total += H;  // weight
    
    return total;
}

