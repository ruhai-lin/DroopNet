/**
 * @file main.c
 * @brief TinyGRU INT8 inference test program
 * 
 * Functions:
 * 1. Load model weights
 * 2. Load test data
 * 3. Run inference
 * 4. Compute F1 Score
 * 5. Measure Latency/Throughput
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "model.h"
#include "inference.h"

/* ========================================
 *          Configuration
 * ======================================== */

#define WEIGHT_PATH     "../outputs/tiny_gru_int8.bin"
#define DATA_PATH       "../../pdn_dataset_uint8.bin"

// Performance benchmark parameters
#define WARMUP_RUNS     10
#define BENCHMARK_RUNS  1000

/* ========================================
 *          Data loading
 * ======================================== */

/**
 * @brief Load test data from .bin file
 * 
 * File format:
 * - Header: Magic(0xAABBCCDD), N, W, C (4 bytes each)
 * - Data: repeat N times [X: W*C bytes, y: 1 byte]
 */
typedef struct {
    uint8_t *X;     // [N, W, C]
    uint8_t *y;     // [N]
    int n_samples;
    int window_size;
    int n_channels;
} TestDataset;

int load_test_data(TestDataset *data, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[Error] Cannot open data file: %s\n", filepath);
        return -1;
    }
    
    // Read header
    uint32_t magic, n, w, c;
    if (fread(&magic, 4, 1, f) != 1 ||
        fread(&n, 4, 1, f) != 1 ||
        fread(&w, 4, 1, f) != 1 ||
        fread(&c, 4, 1, f) != 1) {
        fprintf(stderr, "[Error] Failed to read header\n");
        fclose(f);
        return -1;
    }
    
    if (magic != 0xAABBCCDD) {
        fprintf(stderr, "[Error] Invalid magic number: 0x%08X\n", magic);
        fclose(f);
        return -1;
    }
    
    printf("[Data] Loading %u samples (W=%u, C=%u)...\n", n, w, c);
    
    data->n_samples = (int)n;
    data->window_size = (int)w;
    data->n_channels = (int)c;
    
    // Allocate memory
    int sample_size = w * c;
    data->X = (uint8_t *)malloc(n * sample_size);
    data->y = (uint8_t *)malloc(n);
    
    if (!data->X || !data->y) {
        fprintf(stderr, "[Error] Memory allocation failed\n");
        fclose(f);
        return -1;
    }
    
    // Read data
    for (uint32_t i = 0; i < n; i++) {
        if (fread(data->X + i * sample_size, 1, sample_size, f) != (size_t)sample_size) {
            fprintf(stderr, "[Error] Failed to read sample %u X\n", i);
            fclose(f);
            return -1;
        }
        if (fread(data->y + i, 1, 1, f) != 1) {
            fprintf(stderr, "[Error] Failed to read sample %u y\n", i);
            fclose(f);
            return -1;
        }
    }
    
    fclose(f);
    printf("[Data] Loaded successfully\n");
    return 0;
}

void free_test_data(TestDataset *data) {
    if (data->X) { free(data->X); data->X = NULL; }
    if (data->y) { free(data->y); data->y = NULL; }
}

/* ========================================
 *          Evaluation metrics
 * ======================================== */

typedef struct {
    int tp;     // True Positive
    int fp;     // False Positive
    int tn;     // True Negative
    int fn;     // False Negative
} ConfusionMatrix;

void compute_confusion_matrix(ConfusionMatrix *cm,
                              const float *outputs,
                              const uint8_t *labels,
                              int n_samples) {
    cm->tp = cm->fp = cm->tn = cm->fn = 0;
    
    for (int i = 0; i < n_samples; i++) {
        // Sigmoid
        float prob = 1.0f / (1.0f + expf(-outputs[i]));
        int pred = (prob > 0.5f) ? 1 : 0;
        int label = (int)labels[i];
        
        if (pred == 1 && label == 1) cm->tp++;
        else if (pred == 1 && label == 0) cm->fp++;
        else if (pred == 0 && label == 0) cm->tn++;
        else cm->fn++;
    }
}

float compute_accuracy(const ConfusionMatrix *cm) {
    int total = cm->tp + cm->fp + cm->tn + cm->fn;
    return total > 0 ? (float)(cm->tp + cm->tn) / (float)total : 0.0f;
}

float compute_precision(const ConfusionMatrix *cm) {
    int denom = cm->tp + cm->fp;
    return denom > 0 ? (float)cm->tp / (float)denom : 0.0f;
}

float compute_recall(const ConfusionMatrix *cm) {
    int denom = cm->tp + cm->fn;
    return denom > 0 ? (float)cm->tp / (float)denom : 0.0f;
}

float compute_f1(const ConfusionMatrix *cm) {
    float prec = compute_precision(cm);
    float rec = compute_recall(cm);
    return (prec + rec) > 0 ? 2.0f * prec * rec / (prec + rec) : 0.0f;
}

/* ========================================
 *          Performance measurement
 * ======================================== */

double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ========================================
 *          Main function
 * ======================================== */

int main(int argc, char *argv[]) {
    const char *weight_path = WEIGHT_PATH;
    const char *data_path = DATA_PATH;
    
    // Override via command line args
    if (argc > 1) weight_path = argv[1];
    if (argc > 2) data_path = argv[2];
    
    printf("============================================================\n");
    printf("   TinyGRU INT8 C Model\n");
    printf("============================================================\n\n");
    
    // 1. Load model
    printf("[1] Loading Model...\n");
    TinyGRUModel model;
    if (model_load(&model, weight_path) < 0) {
        return 1;
    }
    model_print_info(&model);
    
    // 2. Initialize inference buffer
    printf("[2] Initializing Inference Buffer...\n");
    InferenceBuffer buf;
    if (inference_buffer_init(&buf, model.hidden_size) < 0) {
        fprintf(stderr, "[Error] Failed to init inference buffer\n");
        model_free(&model);
        return 1;
    }
    printf("    Hidden state: %d bytes\n", model.hidden_size);
    printf("    Acc buffer: %zu bytes\n", 3 * model.hidden_size * sizeof(int32_t));
    printf("    Gate buffer: %zu bytes\n\n", 3 * model.hidden_size * sizeof(float));
    
    // 3. Load test data
    printf("[3] Loading Test Data...\n");
    TestDataset data;
    if (load_test_data(&data, data_path) < 0) {
        inference_buffer_free(&buf);
        model_free(&model);
        return 1;
    }
    
    // Validate data dimensions
    if (data.window_size != WINDOW_SIZE || data.n_channels != INPUT_SIZE) {
        fprintf(stderr, "[Error] Data dimension mismatch: got (%d, %d), expected (%d, %d)\n",
                data.window_size, data.n_channels, WINDOW_SIZE, INPUT_SIZE);
        free_test_data(&data);
        inference_buffer_free(&buf);
        model_free(&model);
        return 1;
    }
    printf("    Samples: %d\n\n", data.n_samples);
    
    // 4. Run inference and compute accuracy
    printf("[4] Running Inference...\n");
    int sample_size = data.window_size * data.n_channels;
    float *outputs = (float *)malloc(data.n_samples * sizeof(float));
    
    if (!outputs) {
        fprintf(stderr, "[Error] Memory allocation failed\n");
        free_test_data(&data);
        inference_buffer_free(&buf);
        model_free(&model);
        return 1;
    }
    
    // Inference for all samples
    for (int i = 0; i < data.n_samples; i++) {
        outputs[i] = inference_forward(&model, &buf, data.X + i * sample_size);
    }
    
    // Compute evaluation metrics
    ConfusionMatrix cm;
    compute_confusion_matrix(&cm, outputs, data.y, data.n_samples);
    
    float accuracy = compute_accuracy(&cm);
    float precision = compute_precision(&cm);
    float recall = compute_recall(&cm);
    float f1 = compute_f1(&cm);
    
    printf("\n------------------------------------------------------------\n");
    printf("                    Evaluation Results\n");
    printf("------------------------------------------------------------\n");
    printf("  Confusion Matrix:\n");
    printf("                  Predicted\n");
    printf("                  Neg    Pos\n");
    printf("    Actual Neg    %4d   %4d\n", cm.tn, cm.fp);
    printf("           Pos    %4d   %4d\n", cm.fn, cm.tp);
    printf("\n");
    printf("  Accuracy:  %.4f (%d/%d)\n", accuracy, cm.tp + cm.tn, data.n_samples);
    printf("  Precision: %.4f\n", precision);
    printf("  Recall:    %.4f\n", recall);
    printf("  F1 Score:  %.4f\n", f1);
    printf("------------------------------------------------------------\n\n");
    
    // 5. Performance test
    printf("[5] Performance Benchmark...\n");
    printf("    Warmup: %d runs\n", WARMUP_RUNS);
    printf("    Benchmark: %d runs\n", BENCHMARK_RUNS);
    
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        inference_forward(&model, &buf, data.X);
    }
    
    // Benchmark
    double start_us = get_time_us();
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        inference_forward(&model, &buf, data.X + (i % data.n_samples) * sample_size);
    }
    double end_us = get_time_us();
    
    double total_us = end_us - start_us;
    double latency_us = total_us / BENCHMARK_RUNS;
    double throughput = 1e6 / latency_us;  // samples/sec
    
    printf("\n------------------------------------------------------------\n");
    printf("                    Performance Results\n");
    printf("------------------------------------------------------------\n");
    printf("  Total time:    %.2f ms (%d samples)\n", total_us / 1000.0, BENCHMARK_RUNS);
    printf("  Latency:       %.2f us/sample\n", latency_us);
    printf("  Throughput:    %.0f samples/sec\n", throughput);
    printf("  Throughput:    %.2f KOPS\n", throughput / 1000.0);
    printf("------------------------------------------------------------\n\n");
    
    // 6. ASIC reference info
    printf("[6] ASIC Reference Info\n");
    printf("------------------------------------------------------------\n");
    size_t weight_bytes = model_weight_bytes(&model);
    size_t activation_bytes = model.hidden_size + WINDOW_SIZE * model.input_size;
    printf("  Weight Size:      %zu bytes (%.2f KB)\n", weight_bytes, weight_bytes / 1024.0);
    printf("  Activation Est:   %zu bytes\n", activation_bytes);
    printf("  Total SRAM Est:   %zu bytes (%.2f KB)\n", 
           weight_bytes + activation_bytes, (weight_bytes + activation_bytes) / 1024.0);
    printf("  6KB Budget Fit:   %s\n", 
           (weight_bytes + activation_bytes) <= 6144 ? "YES" : "NO");
    printf("  8KB SRAM Fit:     %s\n", 
           (weight_bytes + activation_bytes) <= 8192 ? "YES" : "NO");
    printf("------------------------------------------------------------\n\n");
    
    // 7. Output sample results for comparison/verification
    printf("[7] Sample Outputs (for verification)\n");
    printf("------------------------------------------------------------\n");
    printf("  Sample  Label  Output     Prob    Pred\n");
    printf("------------------------------------------------------------\n");
    for (int i = 0; i < 10 && i < data.n_samples; i++) {
        float prob = 1.0f / (1.0f + expf(-outputs[i]));
        int pred = (prob > 0.5f) ? 1 : 0;
        printf("  %4d    %d      %8.4f   %.4f  %d\n", 
               i, data.y[i], outputs[i], prob, pred);
    }
    printf("------------------------------------------------------------\n\n");
    
    // Cleanup
    free(outputs);
    free_test_data(&data);
    inference_buffer_free(&buf);
    model_free(&model);
    
    printf("============================================================\n");
    printf("   Done!\n");
    printf("============================================================\n");
    
    return 0;
}

