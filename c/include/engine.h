/**
 * RegesARC Engine Core — Internal header.
 *
 * Implements:
 *   - GGUF weight loading (header + tensor metadata)
 *   - Expert residency cache with LRU eviction per layer
 *   - Disk streaming for cold experts (SSD-backed)
 *   - Speculative decoding (MTP draft + verify)
 *   - Thread-safe model lifecycle
 */

#ifndef REGES_ENGINE_H
#define REGES_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── GGUF magic and constants ──────────────────────────────────────── */
#define REGES_GGUF_MAGIC 0x46554747  /* "GGUF" in little-endian */
#define REGES_MAX_TENSORS 8192
#define REGES_MAX_PARAMS  256

/* ── Tensor types (subset of GGUF) ─────────────────────────────────── */
typedef enum {
    REGES_TENSOR_F32     = 0,
    REGES_TENSOR_F16     = 1,
    REGES_TENSOR_Q4_0    = 2,
    REGES_TENSOR_Q4_1    = 3,
    REGES_TENSOR_Q5_0    = 6,
    REGES_TENSOR_Q5_1    = 7,
    REGES_TENSOR_Q8_0    = 8,
    REGES_TENSOR_IQ2_XXS = 9,
    REGES_TENSOR_IQ3_XXS = 11,
} reges_tensor_type_t;

/* ── Tensor metadata (from GGUF header) ───────────────────────────── */
typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];       /* [rows, cols, depth, pad] */
    reges_tensor_type_t type;
    uint64_t offset;        /* byte offset in file */
    uint64_t size_bytes;
} reges_tensor_meta_t;

/* ── Model config (extracted from GGUF) ───────────────────────────── */
typedef struct {
    char family[64];
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t hidden_dim;
    uint32_t n_experts;
    uint32_t vocab_size;
    uint32_t n_kv_heads;
    float    rope_freq_base;
    float    rope_freq_scale;
    bool     has_mtp_heads;   /* speculative decoding support */
} reges_model_config_t;

/* ── Expert residency cache entry ─────────────────────────────────── */
typedef struct {
    uint32_t expert_id;
    uint64_t last_access;   /* monotonic clock ticks */
    bool     pinned;        /* never evicted unless explicitly unpinned */
} reges_expert_entry_t;

/* ── Per-layer residency cache ────────────────────────────────────── */
typedef struct {
    uint32_t capacity;          /* max experts resident */
    uint32_t current_count;     /* currently resident */
    uint64_t resident_bytes;    /* total bytes resident */
    reges_expert_entry_t *entries;  /* dynamic array */
    uint64_t clock;             /* monotonic access counter */
} reges_layer_cache_t;

/* ── Model state (opaque to Python) ───────────────────────────────── */
typedef struct {
    /* Metadata */
    reges_model_config_t config;
    reges_tensor_meta_t  tensors[REGES_MAX_TENSORS];
    uint32_t n_tensors;

    /* Weight storage (simplified: mmap or in-RAM) */
    uint8_t *weight_data;     /* base pointer for loaded weights */
    size_t   weight_size;     /* total bytes mapped/allocated */

    /* Expert residency per layer */
    reges_layer_cache_t *layer_caches;  /* [n_layers] */
    uint32_t n_layer_caches;

    /* Disk streaming state */
    FILE      *stream_file;       /* open GGUF for cold expert reads */
    char       stream_path[512];
    bool       stream_enabled;

    /* Speculative decoding */
    bool        spec_decode_enabled;
    uint32_t    n_draft_tokens;   /* MTP draft length */
    float       spec_temperature;

    /* Performance counters */
    uint64_t tokens_generated;
    uint64_t tokens_drafted;
    uint64_t tokens_accepted;
    double   first_token_latency_ms;
    double   throughput_tokens_per_sec;
    size_t   vram_used_bytes;
    size_t   ram_used_bytes;

    /* Thread safety */
    pthread_mutex_t lock;
} reges_model_t;

/* ── Engine API ───────────────────────────────────────────────────── */

/**
 * Load a model from a GGUF file.
 * Returns NULL on failure (sets internal error).
 */
reges_model_t *reges_model_load(const char *gguf_path, const char *adapter_path);

/**
 * Free all resources associated with a loaded model.
 */
void reges_model_free(reges_model_t *model);

/**
 * Get the model configuration (family, layers, heads, etc.).
 */
const reges_model_config_t *reges_model_config(const reges_model_t *model);

/**
 * Pin N hot experts to VRAM/RAM (resident cache).
 * Returns 0 on success.
 */
int reges_expert_pin(reges_model_t *model, uint32_t n_hot);

/**
 * Stream a specific expert from disk into the residency cache.
 * Returns 0 on success.
 */
int reges_expert_stream_in(reges_model_t *model, uint32_t expert_id);

/**
 * Evict an expert back to disk (remove from residency cache).
 * Returns 0 on success.
 */
int reges_expert_evict(reges_model_t *model, uint32_t expert_id);

/**
 * Get the last error message from the engine.
 */
const char *reges_errmsg(void);

#ifdef __cplusplus
}
#endif

#endif /* REGES_ENGINE_H */
