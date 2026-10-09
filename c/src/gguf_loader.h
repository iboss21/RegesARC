#ifndef REGES_GGUF_LOADER_H
#define REGES_GGUF_LOADER_H

/**
 * GGUF weight file loader with mmap support.
 * Compatible with llama.cpp ecosystem GGUF format.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── GGUF magic and version ───────────────────────────────────────────── */
#define GGUF_MAGIC 0x46554747  /* "GGUF" in little-endian */
#define GGUF_VERSION_3 3

/* ── Tensor data types (subset of llama.cpp) ──────────────────────────── */
typedef enum {
    GGUF_TYPE_F32     = 0,
    GGUF_TYPE_F16     = 1,
    GGUF_TYPE_Q4_0    = 2,
    GGUF_TYPE_Q4_1    = 3,
    GGUF_TYPE_Q5_0    = 6,
    GGUF_TYPE_Q5_1    = 7,
    GGUF_TYPE_Q8_0    = 8,
    GGUF_TYPE_I8      = 9,
    GGUF_TYPE_I16     = 10,
    GGUF_TYPE_I32     = 11,
    GGUF_TYPE_I64     = 12,
    GGUF_TYPE_F64     = 13,
    GGUF_TYPE_Q8_1    = 15,
    GGUF_TYPE_BOOL    = 17,
    GGUF_TYPE_U8      = 19,
} gguf_type_t;

/* ── Metadata value types ─────────────────────────────────────────────── */
typedef enum {
    GGUF_VAL_TYPE_UINT8   = 0,
    GGUF_VAL_TYPE_INT8    = 1,
    GGUF_VAL_TYPE_UINT16  = 2,
    GGUF_VAL_TYPE_INT16   = 3,
    GGUF_VAL_TYPE_UINT32  = 4,
    GGUF_VAL_TYPE_INT32   = 5,
    GGUF_VAL_TYPE_FLOAT32 = 6,
    GGUF_VAL_TYPE_BOOL    = 7,
    GGUF_VAL_TYPE_STRING  = 8,
    GGUF_VAL_TYPE_ARRAY   = 9,
    GGUF_VAL_TYPE_UINT64  = 10,
    GGUF_VAL_TYPE_INT64   = 11,
    GGUF_VAL_TYPE_FLOAT64 = 12,
} gguf_val_type_t;

/* ── GGUF file handle ─────────────────────────────────────────────────── */
typedef struct gguf_file {
    /* File descriptor or mmap pointer */
    void *file_ptr;
    size_t file_size;
    
    /* Header fields */
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    
    /* Metadata key-value pairs (stored as strings) */
    char **kv_keys;
    void  **kv_values;
    gguf_val_type_t *kv_types;
    size_t n_kv_allocated;
    
    /* Tensor info array */
    struct {
        char name[256];
        uint32_t n_dims;
        uint64_t dims[4];
        gguf_type_t type;
        uint64_t offset;      /* Byte offset in file */
        uint64_t size_bytes;  /* Raw size in file (may differ from element count due to quantization) */
    } *tensors;
    
    /* Memory-mapped region (if mmap was used) */
    bool use_mmap;
    void *mmap_ptr;
} gguf_file_t;

/* ── API ──────────────────────────────────────────────────────────────── */

/**
 * Open a GGUF file. Returns NULL on failure.
 * If use_mmap is true, attempts to memory-map the file for zero-copy reads.
 */
gguf_file_t *gguf_open(const char *path, bool use_mmap);

/** Close and free a GGUF file handle. */
void gguf_close(gguf_file_t *f);

/* ── Metadata accessors ───────────────────────────────────────────────── */

uint32_t gguf_get_u32(const gguf_file_t *f, const char *key);
int64_t  gguf_get_i64(const gguf_file_t *f, const char *key);
float    gguf_get_f32(const gguf_file_t *f, const char *key);
bool     gguf_get_bool(const gguf_file_t *f, const char *key);
const char *gguf_get_string(const gguf_file_t *f, const char *key);

/* ── Tensor access ────────────────────────────────────────────────────── */

/** Get the offset and size of a tensor by name. Returns false if not found. */
bool gguf_tensor_info(const gguf_file_t *f, const char *name,
                      uint64_t *offset_out, uint64_t *size_bytes_out);

/** Read raw bytes for a tensor into a buffer. Returns number of bytes read. */
uint64_t gguf_read_tensor(const gguf_file_t *f, const char *name, void *buf, size_t buf_size);

/* ── Weight loading convenience ───────────────────────────────────────── */

/**
 * Load all weights from the GGUF file into a contiguous float32 buffer.
 * Handles dequantization for Q4/Q5/Q8 formats.
 * Returns the number of float32 values loaded, or 0 on error.
 */
size_t gguf_load_all_weights(gguf_file_t *f, float *out_buffer);

/** Get total parameter count (sum of all tensor element counts). */
uint64_t gguf_total_params(const gguf_file_t *f);

#ifdef __cplusplus
}
#endif

#endif /* REGES_GGUF_LOADER_H */
