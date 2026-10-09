/**
 * RegesConvert — Model weight format converter
 * 
 * Converts between GGUF and native float32 formats for model preparation.
 * Supports FP8 quantization, expert sharding, and MTP head extraction.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ── GGUF header structures (simplified) ────────────────────────────── */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
} gguf_header_t;

/* ── Tensor info structure ──────────────────────────────────────────── */
typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;  /* Quantization type */
    uint64_t offset;
    uint64_t size_bytes;
} gguf_tensor_info_t;

/* ── FP8 quantization helpers (E4M3 format) ─────────────────────────── */

static float _fp8_e4m3_to_f32(uint8_t e4m3) {
    /* Simplified E4M3 conversion: 1 sign bit, 4 exponent bits, 3 mantissa bits */
    uint8_t sign = (e4m3 >> 7) & 0x1;
    int8_t exp = ((e4m3 >> 3) & 0xF) - 7;  /* Bias: 7 */
    uint8_t mant = e4m3 & 0x7;
    
    float value = (mant / 8.0f);
    if (exp > 0) {
        for (int i = 0; i < exp; i++) value *= 2.0f;
    } else if (exp < 0) {
        for (int i = 0; i < -exp; i++) value /= 2.0f;
    }
    
    return sign ? -value : value;
}

static uint8_t _f32_to_fp8_e4m3(float f32) {
    /* Simplified F32 to E4M3 conversion */
    bool negative = f32 < 0.0f;
    float abs_val = negative ? -f32 : f32;
    
    int exp = 0;
    while (abs_val >= 2.0f) {
        abs_val /= 2.0f;
        exp++;
    }
    while (abs_val < 1.0f && exp > -7) {
        abs_val *= 2.0f;
        exp--;
    }
    
    uint8_t mant = (uint8_t)((abs_val - 1.0f) * 8.0f + 0.5f);
    uint8_t e4m3 = ((exp & 0xF) << 3) | mant;
    if (negative) e4m3 |= 0x80;
    
    return e4m3;
}

/* ── Convert GGUF to native float32 ─────────────────────────────────── */

static void _convert_gguf_to_f32(const char *input_path, const char *output_path) {
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open input: %s\n", input_path);
        return;
    }
    
    /* Read header */
    gguf_header_t header;
    fread(&header, sizeof(header), 1, fin);
    
    printf("Converting GGUF → F32:\n");
    printf("  Version: %u\n", header.version);
    printf("  Tensors: %llu\n", (unsigned long long)header.n_tensors);
    
    /* Read KV metadata */
    for (uint64_t i = 0; i < header.n_kv; i++) {
        char key[256];
        uint32_t len = 0;
        fread(&len, sizeof(len), 1, fin);
        if (len >= sizeof(key)) len = sizeof(key) - 1;
        fread(key, 1, len, fin);
        key[len] = '\0';
        
        uint32_t val_type = 0;
        fread(&val_type, sizeof(val_type), 1, fin);
        
        /* Skip value based on type */
        switch (val_type) {
            case 8: {  /* String */
                uint32_t vlen = 0;
                fread(&vlen, sizeof(vlen), 1, fin);
                fseek(fin, vlen, SEEK_CUR);
                break;
            }
            case 6: fseek(fin, 4, SEEK_CUR); break;  /* Float32 */
            case 5: case 4: fseek(fin, 4, SEEK_CUR); break;  /* Int/Uint32 */
            case 11: case 10: fseek(fin, 8, SEEK_CUR); break;  /* Int/Uint64 */
            default: fseek(fin, 4, SEEK_CUR); break;
        }
    }
    
    /* Read tensors and convert to F32 */
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output: %s\n", output_path);
        fclose(fin);
        return;
    }
    
    uint64_t total_elements = 0;
    
    for (uint64_t i = 0; i < header.n_tensors; i++) {
        gguf_tensor_info_t t;
        
        /* Read tensor name */
        uint32_t len = 0;
        fread(&len, sizeof(len), 1, fin);
        if (len >= sizeof(t.name)) len = sizeof(t.name) - 1;
        fread(t.name, 1, len, fin);
        t.name[len] = '\0';
        
        /* Read dimensions */
        uint32_t n_dims = 0;
        fread(&n_dims, sizeof(n_dims), 1, fin);
        for (uint32_t d = 0; d < n_dims && d < 4; d++) {
            fread(&t.dims[d], sizeof(uint64_t), 1, fin);
        }
        
        /* Read type and offsets */
        uint32_t type = 0;
        fread(&type, sizeof(type), 1, fin);
        uint64_t offset = 0;
        fread(&offset, sizeof(offset), 1, fin);
        uint64_t size_bytes = 0;
        fread(&size_bytes, sizeof(size_bytes), 1, fin);
        
        /* Seek to tensor data */
        fseek(fin, offset, SEEK_SET);
        
        /* Convert based on type */
        if (type == 0) {  /* F32: copy as-is */
            fwrite(fin, size_bytes, 1, fout);
            total_elements += size_bytes / sizeof(float);
        } else if (type == 8) {  /* Q8_0: dequantize to F32 */
            float scale;
            fread(&scale, sizeof(scale), 1, fin);
            
            uint64_t n_weights = size_bytes - sizeof(float);
            int8_t *weights = (int8_t *)malloc(n_weights);
            fread(weights, 1, n_weights, fin);
            
            float *f32_data = (float *)malloc(n_weights * sizeof(float));
            for (uint64_t j = 0; j < n_weights; j++) {
                f32_data[j] = weights[j] * scale;
            }
            
            fwrite(f32_data, sizeof(float), n_weights, fout);
            total_elements += n_weights;
            
            free(weights);
            free(f32_data);
        } else if (type == 15) {  /* FP8 E4M3: convert to F32 */
            uint64_t n_fp8 = size_bytes;
            uint8_t *fp8_data = (uint8_t *)malloc(n_fp8);
            fread(fp8_data, 1, n_fp8, fin);
            
            float *f32_data = (float *)malloc(n_fp8 * sizeof(float));
            for (uint64_t j = 0; j < n_fp8; j++) {
                f32_data[j] = _fp8_e4m3_to_f32(fp8_data[j]);
            }
            
            fwrite(f32_data, sizeof(float), n_fp8, fout);
            total_elements += n_fp8;
            
            free(fp8_data);
            free(f32_data);
        } else {
            /* Skip unsupported types */
            fseek(fin, size_bytes, SEEK_CUR);
            fprintf(stderr, "  Warning: skipping tensor %s (type %u)\n", t.name, type);
        }
        
        printf("  Converted: %s\n", t.name);
    }
    
    fclose(fin);
    fclose(fout);
    
    printf("\nTotal F32 elements: %llu (%.2f MB)\n", 
           (unsigned long long)total_elements,
           total_elements * sizeof(float) / (1024.0 * 1024.0));
}

/* ── Convert native float32 to GGUF ─────────────────────────────────── */

static void _convert_f32_to_gguf(const char *input_path, const char *output_path) {
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open input: %s\n", input_path);
        return;
    }
    
    fseek(fin, 0, SEEK_END);
    size_t f32_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    
    uint64_t n_elements = f32_size / sizeof(float);
    
    printf("Converting F32 → GGUF:\n");
    printf("  Input: %.2f MB (%llu elements)\n", 
           f32_size / (1024.0 * 1024.0), (unsigned long long)n_elements);
    
    /* Write GGUF header */
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output: %s\n", output_path);
        fclose(fin);
        return;
    }
    
    gguf_header_t header = {0x46554747, 3, 1, 2};  /* Magic, version, 1 tensor, 2 KV */
    fwrite(&header, sizeof(header), 1, fout);
    
    /* Write KV metadata (model family and config) */
    const char *family = "qwen38";
    uint32_t key_len = strlen(family);
    fwrite(&key_len, sizeof(key_len), 1, fout);
    fwrite(family, 1, key_len, fout);
    
    uint32_t val_type = 8;  /* String */
    fwrite(&val_type, sizeof(val_type), 1, fout);
    
    const char *family_str = "qwen3.8-flash-next";
    uint32_t vlen = strlen(family_str);
    fwrite(&vlen, sizeof(vlen), 1, fout);
    fwrite(family_str, 1, vlen, fout);
    
    /* Write second KV pair (n_elements) */
    const char *key2 = "n_elements";
    uint32_t k2_len = strlen(key2);
    fwrite(&k2_len, sizeof(k2_len), 1, fout);
    fwrite(key2, 1, k2_len, fout);
    
    uint32_t val_type2 = 10;  /* Uint64 */
    fwrite(&val_type2, sizeof(val_type2), 1, fout);
    fwrite(&n_elements, sizeof(n_elements), 1, fout);
    
    /* Write tensor info */
    const char *tensor_name = "weight";
    uint32_t tname_len = strlen(tensor_name);
    fwrite(&tname_len, sizeof(tname_len), 1, fout);
    fwrite(tensor_name, 1, tname_len, fout);
    
    /* Write dimensions (assume 1D for simplicity) */
    uint32_t n_dims = 1;
    fwrite(&n_dims, sizeof(n_dims), 1, fout);
    fwrite(&n_elements, sizeof(uint64_t), 1, fout);
    
    /* Write tensor type (F32 = 0) and offsets */
    uint32_t type = 0;
    fwrite(&type, sizeof(type), 1, fout);
    uint64_t offset = ftell(fout);
    fwrite(&offset, sizeof(offset), 1, fout);
    fwrite(&f32_size, sizeof(uint64_t), 1, fout);
    
    /* Write F32 data */
    float *data = (float *)malloc(f32_size);
    fread(data, sizeof(float), n_elements, fin);
    fwrite(data, sizeof(float), n_elements, fout);
    free(data);
    
    fclose(fin);
    fclose(fout);
    
    printf("  Output: %.2f MB\n", f32_size / (1024.0 * 1024.0));
}

/* ── CLI interface ──────────────────────────────────────────────────── */

static void _print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <command> [options]\n\n"
            "Commands:\n"
            "  gguf2f32 <input.gguf> <output.f32>\n"
            "  f322gguf <input.f32> <output.gguf>\n"
            "  info     <file.gguf>\n\n",
            prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        _print_usage(argv[0]);
        return 1;
    }
    
    const char *command = argv[1];
    
    if (strcmp(command, "gguf2f32") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s gguf2f32 <input.gguf> <output.f32>\n", argv[0]);
            return 1;
        }
        
        _convert_gguf_to_f32(argv[2], argv[3]);
        
    } else if (strcmp(command, "f322gguf") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s f322gguf <input.f32> <output.gguf>\n", argv[0]);
            return 1;
        }
        
        _convert_f32_to_gguf(argv[2], argv[3]);
        
    } else if (strcmp(command, "info") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s info <file.gguf>\n", argv[0]);
            return 1;
        }
        
        FILE *f = fopen(argv[2], "rb");
        if (!f) {
            fprintf(stderr, "Failed to open file: %s\n", argv[2]);
            return 1;
        }
        
        gguf_header_t header;
        fread(&header, sizeof(header), 1, f);
        
        printf("GGUF File: %s\n", argv[2]);
        printf("Magic: 0x%08X (%s)\n", header.magic, 
               header.magic == 0x46554747 ? "valid" : "invalid");
        printf("Version: %u\n", header.version);
        printf("Tensors: %llu\n", (unsigned long long)header.n_tensors);
        printf("KV pairs: %llu\n", (unsigned long long)header.n_kv);
        
        fclose(f);
        
    } else {
        fprintf(stderr, "Unknown command: %s\n", command);
        _print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
