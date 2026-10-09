/**
 * QPACK — Quantization Packager/Unpacker Utility
 * 
 * Converts between native float32 weights and quantized formats (Q4/Q5/Q8).
 * Used for model conversion and optimization before loading into the engine.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ── Quantization types ─────────────────────────────────────────────── */
typedef enum {
    QPACK_F32 = 0,
    QPACK_F16,
    QPACK_Q4_0,
    QPACK_Q4_1,
    QPACK_Q5_0,
    QPACK_Q5_1,
    QPACK_Q8_0,
} qpack_type_t;

/* ── Quantization block (32 elements per block) ─────────────────────── */
typedef struct {
    float scale;
    int8_t weights[32];  /* Quantized weights */
} qpack_block_t;

/* ── Q4 quantization helpers ────────────────────────────────────────── */

static uint8_t _quantize_q4(float value, float min_val, float max_val) {
    /* Map to 0-15 range (Q4 unsigned) */
    float normalized = (value - min_val) / (max_val - min_val);
    uint8_t quantized = (uint8_t)(normalized * 15.0f + 0.5f);
    return quantized;
}

static float _dequantize_q4(uint8_t q, float min_val, float max_val) {
    /* Map back to original range */
    float normalized = (float)q / 15.0f;
    return min_val + normalized * (max_val - min_val);
}

/* ── Q8 quantization helpers ────────────────────────────────────────── */

static int8_t _quantize_q8(float value, float scale) {
    /* Map to -128 to 127 range (Q8 signed) */
    int8_t quantized = (int8_t)(value / scale + 0.5f);
    return quantized;
}

static float _dequantize_q8(int8_t q, float scale) {
    /* Map back to original range */
    return (float)q * scale;
}

/* ── Pack weights into quantized format ─────────────────────────────── */

static size_t _pack_weights(const float *input, size_t n_elements, 
                            qpack_type_t type, uint8_t *output, size_t output_size) {
    if (type == QPACK_F32) {
        /* No quantization: copy as-is */
        size_t bytes = n_elements * sizeof(float);
        if (bytes > output_size) return 0;
        memcpy(output, input, bytes);
        return bytes;
        
    } else if (type == QPACK_F16) {
        /* Simplified F16: just copy as float32 for now (real impl would convert) */
        size_t bytes = n_elements * sizeof(float);
        if (bytes > output_size) return 0;
        memcpy(output, input, bytes);
        return bytes;
        
    } else if (type == QPACK_Q4_0 || type == QPACK_Q4_1) {
        /* Q4 quantization: 32 elements per block */
        size_t n_blocks = (n_elements + 31) / 32;
        size_t bytes_needed = n_blocks * sizeof(qpack_block_t);
        
        if (bytes_needed > output_size) return 0;
        
        /* Find min/max for each block */
        for (size_t b = 0; b < n_blocks; b++) {
            float block_min = 1e30f, block_max = -1e30f;
            
            for (int i = 0; i < 32 && (b * 32 + i) < n_elements; i++) {
                float val = input[b * 32 + i];
                if (val < block_min) block_min = val;
                if (val > block_max) block_max = val;
            }
            
            qpack_block_t *block = (qpack_block_t *)(output + b * sizeof(qpack_block_t));
            block->scale = (block_max - block_min) / 15.0f;
            
            for (int i = 0; i < 32 && (b * 32 + i) < n_elements; i++) {
                float val = input[b * 32 + i];
                block->weights[i] = _quantize_q4(val, block_min, block_max);
            }
        }
        
        return bytes_needed;
        
    } else if (type == QPACK_Q8_0) {
        /* Q8 quantization: find global scale */
        float max_abs = 0.0f;
        for (size_t i = 0; i < n_elements; i++) {
            float abs_val = input[i] > 0 ? input[i] : -input[i];
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        float scale = max_abs / 127.0f;
        size_t bytes = n_elements * sizeof(int8_t);
        
        if (bytes + sizeof(float) > output_size) return 0;
        
        /* Write scale first */
        memcpy(output, &scale, sizeof(float));
        
        /* Quantize weights */
        int8_t *weights = (int8_t *)(output + sizeof(float));
        for (size_t i = 0; i < n_elements; i++) {
            weights[i] = _quantize_q8(input[i], scale);
        }
        
        return bytes + sizeof(float);
    }
    
    return 0;
}

/* ── Unpack quantized weights back to float32 ───────────────────────── */

static size_t _unpack_weights(const uint8_t *input, size_t input_size,
                              qpack_type_t type, float *output, size_t output_elements) {
    if (type == QPACK_F32 || type == QPACK_F16) {
        /* No quantization: copy as-is */
        size_t bytes = output_elements * sizeof(float);
        if (bytes > input_size) return 0;
        memcpy(output, input, bytes);
        return output_elements;
        
    } else if (type == QPACK_Q4_0 || type == QPACK_Q4_1) {
        /* Q4 dequantization */
        size_t n_blocks = input_size / sizeof(qpack_block_t);
        size_t n_elements = 0;
        
        for (size_t b = 0; b < n_blocks && n_elements < output_elements; b++) {
            const qpack_block_t *block = (const qpack_block_t *)(input + b * sizeof(qpack_block_t));
            
            float block_min = -127.0f * block->scale;
            float block_max = 128.0f * block->scale;
            
            for (int i = 0; i < 32 && n_elements < output_elements; i++) {
                output[n_elements++] = _dequantize_q4(block->weights[i], block_min, block_max);
            }
        }
        
        return n_elements;
        
    } else if (type == QPACK_Q8_0) {
        /* Q8 dequantization */
        float scale;
        memcpy(&scale, input, sizeof(float));
        
        const int8_t *weights = (const int8_t *)(input + sizeof(float));
        size_t n_elements = (input_size - sizeof(float)) / sizeof(int8_t);
        
        if (n_elements > output_elements) n_elements = output_elements;
        
        for (size_t i = 0; i < n_elements; i++) {
            output[i] = _dequantize_q8(weights[i], scale);
        }
        
        return n_elements;
    }
    
    return 0;
}

/* ── CLI interface ──────────────────────────────────────────────────── */

static void _print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <command> [options]\n\n"
            "Commands:\n"
            "  pack   <input.bin> <output.bin> <type>\n"
            "  unpack <input.bin> <output.bin> <type>\n"
            "  info   <file.bin>\n\n"
            "Types: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0\n",
            prog);
}

static qpack_type_t _parse_type(const char *type_str) {
    if (strcmp(type_str, "f32") == 0) return QPACK_F32;
    if (strcmp(type_str, "f16") == 0) return QPACK_F16;
    if (strcmp(type_str, "q4_0") == 0) return QPACK_Q4_0;
    if (strcmp(type_str, "q4_1") == 0) return QPACK_Q4_1;
    if (strcmp(type_str, "q5_0") == 0) return QPACK_Q5_0;
    if (strcmp(type_str, "q5_1") == 0) return QPACK_Q5_1;
    if (strcmp(type_str, "q8_0") == 0) return QPACK_Q8_0;
    
    fprintf(stderr, "Unknown type: %s\n", type_str);
    return QPACK_F32;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        _print_usage(argv[0]);
        return 1;
    }
    
    const char *command = argv[1];
    
    if (strcmp(command, "pack") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s pack <input.bin> <output.bin> <type>\n", argv[0]);
            return 1;
        }
        
        const char *input_path = argv[2];
        const char *output_path = argv[3];
        qpack_type_t type = _parse_type(argv[4]);
        
        /* Read input file */
        FILE *fin = fopen(input_path, "rb");
        if (!fin) {
            fprintf(stderr, "Failed to open input: %s\n", input_path);
            return 1;
        }
        
        fseek(fin, 0, SEEK_END);
        size_t input_size = ftell(fin);
        fseek(fin, 0, SEEK_SET);
        
        float *input_data = (float *)malloc(input_size);
        fread(input_data, sizeof(float), input_size / sizeof(float), fin);
        fclose(fin);
        
        size_t n_elements = input_size / sizeof(float);
        
        /* Pack weights */
        uint8_t *output_data = (uint8_t *)malloc(input_size);  /* Worst case: same size */
        size_t output_bytes = _pack_weights(input_data, n_elements, type, 
                                            output_data, input_size);
        
        if (output_bytes == 0) {
            fprintf(stderr, "Packing failed\n");
            free(input_data);
            free(output_data);
            return 1;
        }
        
        /* Write output file */
        FILE *fout = fopen(output_path, "wb");
        if (!fout) {
            fprintf(stderr, "Failed to open output: %s\n", output_path);
            free(input_data);
            free(output_data);
            return 1;
        }
        
        fwrite(output_data, 1, output_bytes, fout);
        fclose(fout);
        
        printf("Packed %zu elements (%.2f MB) → %.2f MB (%s)\n",
               n_elements, input_size / (1024.0 * 1024.0),
               output_bytes / (1024.0 * 1024.0), argv[4]);
        
        free(input_data);
        free(output_data);
        
    } else if (strcmp(command, "unpack") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s unpack <input.bin> <output.bin> <type>\n", argv[0]);
            return 1;
        }
        
        const char *input_path = argv[2];
        const char *output_path = argv[3];
        qpack_type_t type = _parse_type(argv[4]);
        
        /* Read input file */
        FILE *fin = fopen(input_path, "rb");
        if (!fin) {
            fprintf(stderr, "Failed to open input: %s\n", input_path);
            return 1;
        }
        
        fseek(fin, 0, SEEK_END);
        size_t input_size = ftell(fin);
        fseek(fin, 0, SEEK_SET);
        
        uint8_t *input_data = (uint8_t *)malloc(input_size);
        fread(input_data, 1, input_size, fin);
        fclose(fin);
        
        /* Unpack weights */
        float *output_data = (float *)malloc(input_size);  /* Worst case: same size */
        size_t n_elements = _unpack_weights(input_data, input_size, type, 
                                            output_data, input_size / sizeof(float));
        
        if (n_elements == 0) {
            fprintf(stderr, "Unpacking failed\n");
            free(input_data);
            free(output_data);
            return 1;
        }
        
        /* Write output file */
        FILE *fout = fopen(output_path, "wb");
        if (!fout) {
            fprintf(stderr, "Failed to open output: %s\n", output_path);
            free(input_data);
            free(output_data);
            return 1;
        }
        
        fwrite(output_data, sizeof(float), n_elements, fout);
        fclose(fout);
        
        printf("Unpacked %.2f MB (%s) → %zu elements (%.2f MB)\n",
               input_size / (1024.0 * 1024.0), argv[4],
               n_elements, n_elements * sizeof(float) / (1024.0 * 1024.0));
        
        free(input_data);
        free(output_data);
        
    } else if (strcmp(command, "info") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s info <file.bin>\n", argv[0]);
            return 1;
        }
        
        const char *path = argv[2];
        
        FILE *f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "Failed to open file: %s\n", path);
            return 1;
        }
        
        fseek(f, 0, SEEK_END);
        size_t size = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        printf("File: %s\n", path);
        printf("Size: %.2f MB (%zu bytes)\n", size / (1024.0 * 1024.0), size);
        
        /* Try to detect format from file signature */
        uint8_t header[16];
        fread(header, 1, sizeof(header), f);
        
        if (header[0] == 0x47 && header[1] == 0x47 && header[2] == 0x55 && header[3] == 0x46) {
            printf("Format: GGUF\n");
        } else {
            printf("Format: Unknown (raw binary)\n");
        }
        
        fclose(f);
        
    } else {
        fprintf(stderr, "Unknown command: %s\n", command);
        _print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
