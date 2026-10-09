#ifndef REGES_ENGINE_H
#define REGES_ENGINE_H

/**
 * RegesARC Engine Core — model loading, KV cache, expert routing, spec decode.
 *
 * This is the single binary entry point. Family adapters (qwen38, deepseek_v4,
 * glm, regescore) are loaded as shared libraries at runtime via dlopen/dlsym.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef _WIN32
    #include <windows.h>
    #define REGES_DL_SYM LoadLibraryA
#else
    #include <pthread.h>
    #include <dlfcn.h>
    #define REGES_DL_SYM dlopen
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Forward declarations ─────────────────────────────────────────────── */
struct reges_model;
struct reges_kv_cache;
struct reges_expert_pool;
struct reges_spec_draft;

/* ── Model config (from GGUF metadata) ────────────────────────────────── */
typedef struct {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;       /* 1 for MLA families */
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t n_experts;
    uint32_t n_activated_experts; /* top-k routing */
    bool has_mtp_heads;          /* multi-token prediction for spec decode */
    uint32_t mtp_head_count;     /* number of draft heads */
    char family[64];             /* "qwen38", "deepseek_v4", etc. */
    float rope_scale;
    float rope_theta;
} reges_model_config_t;

/* ── Placement verdict (mirrors Python place.py) ───────────────────────── */
typedef enum {
    REGES_VERDICT_REFUSE = 0,
    REGES_VERDICT_DISK   = 1,
    REGES_VERDICT_CARD   = 2,
} reges_verdict_t;

/* ── Backend type ─────────────────────────────────────────────────────── */
typedef enum {
    REGES_BACKEND_NONE = 0,
    REGES_BACKEND_CUDA,
    REGES_BACKEND_HIP,
} reges_backend_t;

#ifdef _WIN32
#define REGES_BACKEND_CPU REGES_BACKEND_NONE
#endif

/* ── Logging callback (set by host) ───────────────────────────────────── */
typedef void (*reges_log_fn)(int level, const char *msg);
void reges_set_logger(reges_log_fn fn);

/* ════════════════════════════════════════════════════════════════════════ */
/*  MODEL LIFECYCLE                                                         */
/* ════════════════════════════════════════════════════════════════════════ */

/**
 * Load a model from a GGUF file. Returns NULL on failure; call reges_errmsg()
 * for the reason. The returned struct owns all memory until reges_model_free().
 */
struct reges_model *reges_model_load(const char *gguf_path,
                                     const char *family_adapter_path);

/** Free a loaded model and all its resources. */
void reges_model_free(struct reges_model *m);

/** Get the config for a loaded model. */
const reges_model_config_t *reges_model_config(const struct reges_model *m);

/* ════════════════════════════════════════════════════════════════════════ */
/*  KV CACHE                                                                */
/* ════════════════════════════════════════════════════════════════════════ */

/**
 * Allocate a KV cache for `batch_size` sequences, each up to `max_seq_len`
 * tokens. The cache is pre-allocated in GPU VRAM (card path) or pinned RAM
 * (disk path).
 */
struct reges_kv_cache *reges_kv_cache_new(struct reges_model *m,
                                          uint32_t batch_size,
                                          uint32_t max_seq_len);

/** Free a KV cache. */
void reges_kv_cache_free(struct reges_kv_cache *cache);

/** Reset all sequences in the cache (e.g., after a prompt is consumed). */
void reges_kv_cache_reset(struct reges_kv_cache *cache, uint32_t seq_id);

/* ════════════════════════════════════════════════════════════════════════ */
/*  EXPERT RESIDENCY (hot/cold pinning)                                     */
/* ════════════════════════════════════════════════════════════════════════ */

/**
 * Pin `n_hot` experts to GPU VRAM. The remaining experts stay on disk and are
 * streamed in on demand when routed by the MoE router. Returns 0 on success,
 * -1 on OOM or I/O error.
 */
int reges_expert_pin(struct reges_model *m, uint32_t n_hot);

/** Stream a specific expert from disk into VRAM/RAM. */
int reges_expert_stream_in(struct reges_model *m, uint32_t expert_id);

/** Evict an expert back to disk to free memory. */
int reges_expert_evict(struct reges_model *m, uint32_t expert_id);

/* ════════════════════════════════════════════════════════════════════════ */
/*  INFERENCE (single token decode)                                       */
/* ════════════════════════════════════════════════════════════════════════ */

/**
 * Decode one token. `tokens` is an array of batch_size token IDs (the prompt
 * or previous draft tokens). `positions` are the absolute positions in each
 * sequence. Writes logits into `logits_out` (vocab_size × batch_size).
 * Returns 0 on success, -1 on error.
 */
int reges_decode(struct reges_model *m,
                 struct reges_kv_cache *cache,
                 const uint32_t *tokens,
                 const uint32_t *positions,
                 uint32_t batch_size,
                 float *logits_out);

/**
 * Sample from logits. `temperature` > 0 enables sampling; 0 = greedy argmax.
 * Returns the sampled token ID.
 */
uint32_t reges_sample(const float *logits, uint32_t vocab_size,
                      float temperature, uint64_t seed);

/* ════════════════════════════════════════════════════════════════════════ */
/*  SPECULATIVE DECODING (MTP heads)                                      */
/* ════════════════════════════════════════════════════════════════════════ */

/**
 * Generate draft tokens using MTP heads. Returns the number of draft tokens
 * produced (up to `max_draft`). Each draft token is written into `draft_out`.
 * Only valid when model has_mtp_heads == true.
 */
uint32_t reges_spec_generate_draft(struct reges_model *m,
                                   struct reges_kv_cache *cache,
                                   const uint32_t *tokens,
                                   const uint32_t *positions,
                                   uint32_t batch_size,
                                   uint32_t max_draft,
                                   uint32_t *draft_out);

/**
 * Verify draft tokens against the trunk model. Accepts a prefix of the drafts
 * (possibly all, possibly none). Returns the number of accepted tokens.
 */
uint32_t reges_spec_verify(struct reges_model *m,
                           struct reges_kv_cache *cache,
                           const uint32_t *draft_tokens,
                           uint32_t n_draft,
                           float temperature);

/* ════════════════════════════════════════════════════════════════════════ */
/*  ERROR HANDLING                                                          */
/* ════════════════════════════════════════════════════════════════════════ */

/** Return the last error message (thread-safe, static buffer). */
const char *reges_errmsg(void);

#ifdef __cplusplus
}
#endif

#endif /* REGES_ENGINE_H */
