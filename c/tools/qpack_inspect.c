/**
 * QPACK Inspect — Model weight inspection utility
 * 
 * Reads GGUF files and displays model architecture, parameter counts,
 * quantization info, and memory requirements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

/* ── Model info structure ───────────────────────────────────────────── */
typedef struct {
    char family[64];
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t n_experts;
    uint32_t n_activated_experts;
    bool has_mtp;
    uint32_t mtp_head_count;
} model_info_t;

/* ── Read helpers ───────────────────────────────────────────────────── */

static uint32_t read_u32(FILE *f) {
    uint32_t val;
    fread(&val, sizeof(val), 1, f);
    return val;
}

static uint64_t read_u64(FILE *f) {
    uint64_t val;
    fread(&val, sizeof(val), 1, f);
    return val;
}

static void read_string(FILE *f, char *buf, size_t max_len) {
    uint32_t len = read_u32(f);
    if (len >= max_len) len = max_len - 1;
    fread(buf, 1, len, f);
    buf[len] = '\0';
}

/* ── Format size in human-readable form ─────────────────────────────── */

static const char *_format_size(uint64_t bytes) {
    static char buf[64];
    
    if (bytes >= 1024ULL * 1024 * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024 * 1024 * 1024));
    } else if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024 * 1024));
    } else if (bytes >= 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%.2f KB", bytes / (1024.0 * 1024));
    } else {
        snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    }
    
    return buf;
}

/* ── Print tensor info ──────────────────────────────────────────────── */

static void _print_tensor(const gguf_tensor_info_t *t) {
    printf("  %-50s ", t->name);
    
    /* Print dimensions */
    printf("[");
    for (uint32_t d = 0; d < t->n_dims && d < 4; d++) {
        if (d > 0) printf("x");
        printf("%llu", (unsigned long long)t->dims[d]);
    }
    printf("] ");
    
    /* Print size */
    uint64_t total_elements = 1;
    for (uint32_t d = 0; d < t->n_dims && d < 4; d++) {
        total_elements *= t->dims[d];
    }
    
    printf("%-10s ", _format_size(t->size_bytes));
    
    /* Print parameter count */
    char param_buf[64];
    if (total_elements >= 1000000) {
        snprintf(param_buf, sizeof(param_buf), "%.2fM", total_elements / 1000000.0);
    } else {
        snprintf(param_buf, sizeof(param_buf), "%llu", (unsigned long long)total_elements);
    }
    printf("%-8s ", param_buf);
    
    /* Print quantization type */
    const char *qtype = "F32";
    switch (t->type) {
        case 0: qtype = "F32"; break;
        case 1: qtype = "F16"; break;
        case 2: qtype = "Q4_0"; break;
        case 3: qtype = "Q4_1"; break;
        case 6: qtype = "Q5_0"; break;
        case 7: qtype = "Q5_1"; break;
        case 8: qtype = "Q8_0"; break;
        default: qtype = "UNK"; break;
    }
    printf("%s\n", qtype);
}

/* ── Inspect GGUF file ──────────────────────────────────────────────── */

static void _inspect_gguf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return;
    }
    
    /* Read header */
    gguf_header_t header;
    fread(&header, sizeof(header), 1, f);
    
    printf("GGUF File: %s\n", path);
    printf("Version: %u\n", header.version);
    printf("Tensors: %llu\n", (unsigned long long)header.n_tensors);
    printf("KV pairs: %llu\n\n", (unsigned long long)header.n_kv);
    
    /* Read KV metadata */
    printf("Metadata:\n");
    for (uint64_t i = 0; i < header.n_kv; i++) {
        char key[256];
        read_string(f, key, sizeof(key));
        
        uint32_t val_type = read_u32(f);
        
        /* Read value based on type */
        if (val_type == 8) {  /* String */
            char val[1024];
            read_string(f, val, sizeof(val));
            printf("  %-40s = %s\n", key, val);
        } else if (val_type == 6) {  /* Float32 */
            float val;
            fread(&val, sizeof(val), 1, f);
            printf("  %-40s = %.6f\n", key, val);
        } else if (val_type == 5 || val_type == 4) {  /* Int32/Uint32 */
            uint32_t val;
            fread(&val, sizeof(val), 1, f);
            printf("  %-40s = %u\n", key, val);
        } else if (val_type == 11 || val_type == 10) {  /* Int64/Uint64 */
            uint64_t val;
            fread(&val, sizeof(val), 1, f);
            printf("  %-40s = %llu\n", key, (unsigned long long)val);
        } else if (val_type == 7) {  /* Bool */
            uint8_t val;
            fread(&val, 1, 1, f);
            printf("  %-40s = %s\n", key, val ? "true" : "false");
        } else {
            /* Skip unknown types (read and discard) */
            uint64_t skip_size = 0;
            switch (val_type) {
                case 0: skip_size = 1; break;  /* Uint8 */
                case 1: skip_size = 1; break;  /* Int8 */
                case 2: skip_size = 2; break;  /* Uint16 */
                case 3: skip_size = 2; break;  /* Int16 */
                case 4: skip_size = 4; break;  /* Uint32 */
                case 5: skip_size = 4; break;  /* Int32 */
                case 6: skip_size = 4; break;  /* Float32 */
                case 7: skip_size = 1; break;  /* Bool */
                case 9: skip_size = 8; break;  /* Array header */
                case 10: skip_size = 8; break; /* Uint64 */
                case 11: skip_size = 8; break; /* Int64 */
                case 12: skip_size = 8; break; /* Float64 */
                default: skip_size = 4; break;
            }
            fseek(f, skip_size, SEEK_CUR);
            printf("  %-40s = [type %u]\n", key, val_type);
        }
    }
    
    /* Read tensor info */
    printf("\nTensors (%llu total):\n", (unsigned long long)header.n_tensors);
    printf("%-50s %-12s %-8s %s\n", "Name", "Size", "Params", "Type");
    printf("%s\n", "----------------------------------------------------------------------------------");
    
    uint64_t total_params = 0;
    uint64_t total_size = 0;
    
    for (uint64_t i = 0; i < header.n_tensors; i++) {
        gguf_tensor_info_t t;
        
        /* Read tensor name */
        read_string(f, t.name, sizeof(t.name));
        
        /* Read dimensions */
        t.n_dims = read_u32(f);
        for (uint32_t d = 0; d < t.n_dims && d < 4; d++) {
            t.dims[d] = read_u64(f);
        }
        
        /* Read type and offsets */
        t.type = read_u32(f);
        t.offset = read_u64(f);
        t.size_bytes = read_u64(f);
        
        _print_tensor(&t);
        
        /* Accumulate stats */
        uint64_t elem_count = 1;
        for (uint32_t d = 0; d < t.n_dims && d < 4; d++) {
            elem_count *= t.dims[d];
        }
        
        /* Adjust for quantization */
        switch (t.type) {
            case 2: case 3: case 6: case 7: case 8: case 15:
                elem_count /= 32;  /* Q4/Q5/Q8 blocks of 32 */
                break;
            default:
                break;
        }
        
        total_params += elem_count;
        total_size += t.size_bytes;
    }
    
    printf("\nSummary:\n");
    printf("  Total parameters: %.2fM (%llu)\n", 
           total_params / 1000000.0, (unsigned long long)total_params);
    printf("  Total size: %s\n", _format_size(total_size));
    
    fclose(f);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <gguf_file>\n", argv[0]);
        return 1;
    }
    
    _inspect_gguf(argv[1]);
    
    return 0;
}
