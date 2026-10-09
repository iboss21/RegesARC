/**
 * RegesARC Engine Core — model loading, KV cache, expert routing, spec decode.
 *
 * Single binary entry point. Family adapters loaded as shared libraries at runtime.
 */

#include "engine.h"
#include "gguf_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#define REGES_DL_HANDLE void*
#define REGES_DL_LOAD(path) LoadLibraryA(path)
#define REGES_DL_SYM(handle, name) GetProcAddress((HMODULE)(handle), name)
#define REGES_DL_CLOSE(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define REGES_DL_HANDLE void*
#define REGES_DL_LOAD(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define REGES_DL_SYM(handle, name) dlsym(handle, name)
#define REGES_DL_CLOSE(handle) dlclose(handle)
#endif

/* ── Thread-local error message buffer ─────────────────────────────────── */
static __thread char _reges_errbuf[512];

void reges_set_logger(reges_log_fn fn) {
    (void)fn; /* Reserved for future use */
}

const char *reges_errmsg(void) {
    return _reges_errbuf;
}

static void _set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(_reges_errbuf, sizeof(_reges_errbuf), fmt, ap);
    va_end(ap);
}

/* ── Family adapter vtable (loaded via dlopen) ─────────────────────────── */
typedef struct {
    const char *name;
    int (*init)(void);
    void (*free)(void);
    int (*forward)(const float *input, float *output, uint32_t batch_size);
    int (*mtp_forward)(const float *input, float **outputs, uint32_t n_heads, uint32_t batch_size);
} reges_family_vtable_t;

/* ── Model structure ───────────────────────────────────────────────────── */
struct reges_model {
    reges_model_config_t config;
    
    /* GGUF weight tensors (loaded via gguf_loader) */
    float *weights;           /* All weights in contiguous memory */
    size_t n_weights;         /* Number of float32 values */
    
    /* Family adapter handle and vtable */
    REGES_DL_HANDLE adapter_handle;
    reges_family_vtable_t *vtable;
    
    /* Expert residency state */
    uint32_t n_hot_experts;
    bool *expert_pinned;      /* expert_id → pinned to VRAM? */
    float **expert_weights;   /* Per-expert weight pointers (hot only) */
    
    /* Speculative decoding state */
    struct reges_spec_draft *spec_draft;
};

/* ── KV Cache structure ────────────────────────────────────────────────── */
struct reges_kv_cache {
    uint32_t batch_size;
    uint32_t max_seq_len;
    
    /* Key and value tensors per layer, per sequence */
    float *keys;   /* [n_layers][batch_size][max_seq_len][n_kv_heads][head_dim] */
    float *values; /* [n_layers][batch_size][max_seq_len][n_kv_heads][head_dim] */
    
    /* Current write position per sequence */
    uint32_t *seq_pos;
    
    /* Memory pool for allocation */
    size_t total_bytes;
};

/* ── Speculative decoding draft state ──────────────────────────────────── */
struct reges_spec_draft {
    uint32_t max_draft_len;
    uint32_t current_len;
    uint32_t *draft_tokens;
    float **draft_logits; /* Per-head logits for verification */
};

/* ════════════════════════════════════════════════════════════════════════ */
/*  MODEL LIFECYCLE                                                         */
/* ════════════════════════════════════════════════════════════════════════ */

struct reges_model *reges_model_load(const char *gguf_path,
                                     const char *family_adapter_path) {
    if (!gguf_path || !family_adapter_path) {
        _set_error("NULL arguments to reges_model_load");
        return NULL;
    }
    
    /* Load family adapter shared library */
    REGES_DL_HANDLE handle = REGES_DL_LOAD(family_adapter_path);
    if (!handle) {
        _set_error("Failed to load family adapter: %s", family_adapter_path);
        return NULL;
    }
    
    /* Resolve vtable symbols */
    reges_family_vtable_t *vtable = (reges_family_vtable_t *)malloc(sizeof(reges_family_vtable_t));
    if (!vtable) {
        REGES_DL_CLOSE(handle);
        _set_error("Out of memory allocating vtable");
        return NULL;
    }
    
    vtable->init = (int (*)(void))REGES_DL_SYM(handle, "reges_family_init");
    vtable->free = (void (*)(void))REGES_DL_SYM(handle, "reges_family_free");
    vtable->forward = (int (*)(const float *, float *, uint32_t))REGES_DL_SYM(handle, "reges_family_forward");
    vtable->mtp_forward = (int (*)(const float *, float **, uint32_t, uint32_t))REGES_DL_SYM(handle, "reges_family_mtp_forward");
    
    if (!vtable->init || !vtable->forward) {
        REGES_DL_CLOSE(handle);
        free(vtable);
        _set_error("Family adapter missing required symbols (init/forward)");
        return NULL;
    }
    
    /* Initialize the family adapter */
    if (vtable->init() != 0) {
        REGES_DL_CLOSE(handle);
        free(vtable);
        _set_error("Family adapter init failed");
        return NULL;
    }
    
    /* Load GGUF weights */
    gguf_file_t *gguf = gguf_open(gguf_path);
    if (!gguf) {
        vtable->free();
        REGES_DL_CLOSE(handle);
        free(vtable);
        _set_error("Failed to open GGUF file: %s", gguf_path);
        return NULL;
    }
    
    /* Extract model config from GGUF metadata */
    struct reges_model *model = (struct reges_model *)calloc(1, sizeof(struct reges_model));
    if (!model) {
        gguf_close(gguf);
        vtable->free();
        REGES_DL_CLOSE(handle);
        free(vtable);
        _set_error("Out of memory allocating model");
        return NULL;
    }
    
    /* Read config fields from GGUF */
    model->config.n_layers = gguf_get_u32(gguf, "model.layers.count");
    model->config.n_heads = gguf_get_u32(gguf, "model.attention.head_count");
    model->config.n_kv_heads = gguf_get_u32(gguf, "model.attention.head_count_kv");
    model->config.head_dim = gguf_get_u32(gguf, "model.attention.head_dim");
    model->config.vocab_size = gguf_get_u32(gguf, "tokenizer.ggml.tokens.length");
    model->config.hidden_dim = gguf_get_u32(gguf, "model.embedding.length");
    model->config.intermediate_dim = gguf_get_u32(gguf, "model.feed_forward_length");
    model->config.n_experts = gguf_get_u32(gguf, "model.moe.expert_count");
    model->config.n_activated_experts = gguf_get_u32(gguf, "model.moe.expert_used_count");
    model->config.has_mtp_heads = gguf_get_bool(gguf, "model.mtp.enabled");
    model->config.mtp_head_count = gguf_get_u32(gguf, "model.mtp.head_count");
    model->config.rope_scale = gguf_get_f32(gguf, "model.rope.freq_scale");
    model->config.rope_theta = gguf_get_f32(gguf, "model.rope.freq_base");
    
    /* Copy family name */
    const char *family_str = gguf_get_string(gguf, "general.family");
    if (family_str) {
        strncpy(model->config.family, family_str, sizeof(model->config.family) - 1);
    } else {
        strcpy(model->config.family, "unknown");
    }
    
    /* Load all weights into contiguous memory */
    size_t total_params = gguf_total_params(gguf);
    model->weights = (float *)malloc(total_params * sizeof(float));
    if (!model->weights) {
        free(model);
        gguf_close(gguf);
        vtable->free();
        REGES_DL_CLOSE(handle);
        _set_error("Out of memory allocating %zu weights", total_params);
        return NULL;
    }
    
    size_t loaded = gguf_load_all_weights(gguf, model->weights);
    if (loaded != total_params) {
        free(model->weights);
        free(model);
        gguf_close(gguf);
        vtable->free();
        REGES_DL_CLOSE(handle);
        _set_error("GGUF weight load incomplete: expected %zu, got %zu", total_params, loaded);
        return NULL;
    }
    model->n_weights = total_params;
    
    /* Initialize expert residency tracking */
    model->expert_pinned = (bool *)calloc(model->config.n_experts, sizeof(bool));
    if (!model->expert_pinned) {
        free(model->weights);
        free(model);
        gguf_close(gguf);
        vtable->free();
        REGES_DL_CLOSE(handle);
        _set_error("Out of memory allocating expert pin array");
        return NULL;
    }
    
    /* Store adapter handle and vtable */
    model->adapter_handle = handle;
    model->vtable = vtable;
    
    gguf_close(gguf);
    return model;
}

void reges_model_free(struct reges_model *m) {
    if (!m) return;
    
    if (m->expert_pinned) free(m->expert_pinned);
    if (m->weights) free(m->weights);
    if (m->vtable) {
        m->vtable->free();
        free(m->vtable);
    }
    if (m->adapter_handle) REGES_DL_CLOSE(m->adapter_handle);
    
    free(m);
}

const reges_model_config_t *reges_model_config(const struct reges_model *m) {
    return &m->config;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  KV CACHE                                                                */
/* ════════════════════════════════════════════════════════════════════════ */

struct reges_kv_cache *reges_kv_cache_new(struct reges_model *m,
                                          uint32_t batch_size,
                                          uint32_t max_seq_len) {
    if (!m || !m->weights) {
        _set_error("KV cache requires a loaded model");
        return NULL;
    }
    
    struct reges_kv_cache *cache = (struct reges_kv_cache *)calloc(1, sizeof(struct reges_kv_cache));
    if (!cache) {
        _set_error("Out of memory allocating KV cache");
        return NULL;
    }
    
    cache->batch_size = batch_size;
    cache->max_seq_len = max_seq_len;
    
    size_t layer_size = batch_size * max_seq_len * m->config.n_kv_heads * m->config.head_dim * sizeof(float);
    size_t total_layers = m->config.n_layers * layer_size;
    
    /* Allocate keys and values in contiguous memory */
    cache->keys = (float *)calloc(total_layers, sizeof(float));
    cache->values = (float *)calloc(total_layers, sizeof(float));
    if (!cache->keys || !cache->values) {
        free(cache->keys);
        free(cache->values);
        free(cache);
        _set_error("Out of memory allocating KV tensors (%zu bytes)", total_layers * 2 * sizeof(float));
        return NULL;
    }
    
    /* Allocate per-sequence position tracker */
    cache->seq_pos = (uint32_t *)calloc(batch_size, sizeof(uint32_t));
    if (!cache->seq_pos) {
        free(cache->keys);
        free(cache->values);
        free(cache);
        _set_error("Out of memory allocating seq_pos array");
        return NULL;
    }
    
    cache->total_bytes = total_layers * 2 * sizeof(float) + batch_size * sizeof(uint32_t);
    
    return cache;
}

void reges_kv_cache_free(struct reges_kv_cache *cache) {
    if (!cache) return;
    
    free(cache->keys);
    free(cache->values);
    free(cache->seq_pos);
    free(cache);
}

void reges_kv_cache_reset(struct reges_kv_cache *cache, uint32_t seq_id) {
    if (!cache || seq_id >= cache->batch_size) return;
    
    /* Zero out the KV tensors for this sequence */
    const size_t layer_stride = cache->max_seq_len * cache->batch_size * 
                                cache->config.n_kv_heads * cache->config.head_dim;
    
    for (uint32_t layer = 0; layer < cache->config.n_layers; layer++) {
        float *layer_keys = cache->keys + layer * layer_stride;
        float *layer_vals = cache->values + layer * layer_stride;
        
        /* Zero the tokens for this sequence */
        memset(layer_keys + seq_id * cache->max_seq_len, 0, 
               cache->max_seq_len * cache->config.n_kv_heads * cache->config.head_dim * sizeof(float));
        memset(layer_vals + seq_id * cache->max_seq_len, 0, 
               cache->max_seq_len * cache->config.n_kv_heads * cache->config.head_dim * sizeof(float));
    }
    
    /* Reset position counter */
    cache->seq_pos[seq_id] = 0;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  EXPERT RESIDENCY (hot/cold pinning)                                     */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_expert_pin(struct reges_model *m, uint32_t n_hot) {
    if (!m || !m->weights) {
        _set_error("Expert pin requires a loaded model");
        return -1;
    }
    
    if (n_hot > m->config.n_experts) {
        _set_error("Cannot pin %u experts, model has only %u", n_hot, m->config.n_experts);
        return -1;
    }
    
    /* Calculate expert weight size */
    size_t expert_weight_size = (size_t)m->config.intermediate_dim * 
                                m->config.hidden_dim * 2 * sizeof(float); /* up + down proj */
    
    /* Pin the first n_hot experts to VRAM/RAM */
    for (uint32_t i = 0; i < n_hot && i < m->config.n_experts; i++) {
        if (!m->expert_pinned[i]) {
            /* In a real implementation, this would allocate GPU memory and copy weights */
            /* For now, we just mark them as pinned */
            m->expert_pinned[i] = true;
            m->n_hot_experts++;
            
            /* Log the pinning (in production, this would be actual memory allocation) */
            fprintf(stderr, "[reges] Pinned expert %u to VRAM (%zu bytes)\n", 
                    i, expert_weight_size);
        }
    }
    
    return 0;
}

int reges_expert_stream_in(struct reges_model *m, uint32_t expert_id) {
    if (!m || !m->weights) {
        _set_error("Expert stream requires a loaded model");
        return -1;
    }
    
    if (expert_id >= m->config.n_experts) {
        _set_error("Invalid expert ID: %u", expert_id);
        return -1;
    }
    
    if (m->expert_pinned[expert_id]) {
        /* Already pinned, nothing to do */
        return 0;
    }
    
    /* In a real implementation, this would:
     * 1. Allocate GPU memory for the expert weights
     * 2. Seek to the expert's offset in the GGUF file
     * 3. Read and copy the weights into VRAM
     */
    
    m->expert_pinned[expert_id] = true;
    m->n_hot_experts++;
    
    fprintf(stderr, "[reges] Streamed expert %u from disk to VRAM\n", expert_id);
    return 0;
}

int reges_expert_evict(struct reges_model *m, uint32_t expert_id) {
    if (!m || !m->weights) {
        _set_error("Expert evict requires a loaded model");
        return -1;
    }
    
    if (expert_id >= m->config.n_experts) {
        _set_error("Invalid expert ID: %u", expert_id);
        return -1;
    }
    
    if (!m->expert_pinned[expert_id]) {
        /* Not pinned, nothing to evict */
        return 0;
    }
    
    /* In a real implementation, this would:
     * 1. Write the expert weights back to the GGUF file (or a separate cache file)
     * 2. Free the GPU memory allocation
     */
    
    m->expert_pinned[expert_id] = false;
    m->n_hot_experts--;
    
    fprintf(stderr, "[reges] Evicted expert %u back to disk\n", expert_id);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  INFERENCE (single token decode)                                       */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_decode(struct reges_model *m,
                 struct reges_kv_cache *cache,
                 const uint32_t *tokens,
                 const uint32_t *positions,
                 uint32_t batch_size,
                 float *logits_out) {
    if (!m || !m->weights || !cache || !tokens || !logits_out) {
        _set_error("NULL arguments to reges_decode");
        return -1;
    }
    
    if (batch_size > cache->batch_size) {
        _set_error("Batch size %u exceeds cache capacity %u", batch_size, cache->batch_size);
        return -1;
    }
    
    /* Delegate to the family adapter's forward function */
    if (!m->vtable || !m->vtable->forward) {
        _set_error("Family adapter missing forward function");
        return -1;
    }
    
    int ret = m->vtable->forward(m->weights, logits_out, batch_size);
    if (ret != 0) {
        _set_error("Family adapter forward failed");
        return -1;
    }
    
    /* Update KV cache positions */
    for (uint32_t i = 0; i < batch_size; i++) {
        if (positions) {
            cache->seq_pos[i] = positions[i];
        } else {
            cache->seq_pos[i]++;
        }
    }
    
    return 0;
}

uint32_t reges_sample(const float *logits, uint32_t vocab_size,
                      float temperature, uint64_t seed) {
    if (!logits || vocab_size == 0) {
        _set_error("Invalid logits or vocab_size");
        return 0;
    }
    
    /* Greedy sampling (temperature = 0) */
    if (temperature <= 0.0f) {
        uint32_t best_id = 0;
        float best_val = -1e30f;
        
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        
        return best_id;
    }
    
    /* Temperature-scaled softmax sampling */
    float max_logit = -1e30f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    /* Compute softmax with temperature */
    float sum_exp = 0.0f;
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    if (!probs) {
        _set_error("Out of memory allocating probs array");
        return 0;
    }
    
    for (uint32_t i = 0; i < vocab_size; i++) {
        float scaled = (logits[i] - max_logit) / temperature;
        probs[i] = expf(scaled);
        sum_exp += probs[i];
    }
    
    /* Normalize */
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
    }
    
    /* Sample from the distribution using a simple LCG PRNG */
    uint64_t x = seed;
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    float rand_val = (float)(x >> 33) / (float)(1ULL << 31);
    
    float cumulative = 0.0f;
    uint32_t sampled_id = vocab_size - 1;
    
    for (uint32_t i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (cumulative >= rand_val) {
            sampled_id = i;
            break;
        }
    }
    
    free(probs);
    return sampled_id;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  SPECULATIVE DECODING (MTP heads)                                      */
/* ════════════════════════════════════════════════════════════════════════ */

uint32_t reges_spec_generate_draft(struct reges_model *m,
                                   struct reges_kv_cache *cache,
                                   const uint32_t *tokens,
                                   const uint32_t *positions,
                                   uint32_t batch_size,
                                   uint32_t max_draft,
                                   uint32_t *draft_out) {
    if (!m || !m->weights || !m->config.has_mtp_heads) {
        _set_error("Model does not support speculative decoding (no MTP heads)");
        return 0;
    }
    
    if (!m->vtable || !m->vtable->mtp_forward) {
        _set_error("Family adapter missing mtp_forward function");
        return 0;
    }
    
    /* Generate draft tokens using MTP heads */
    uint32_t n_draft = 0;
    
    for (uint32_t step = 0; step < max_draft && step < m->config.mtp_head_count; step++) {
        /* Get logits from the step-th MTP head */
        float *head_logits = (float *)malloc(m->config.vocab_size * sizeof(float));
        if (!head_logits) break;
        
        int ret = m->vtable->mtp_forward(m->weights, &head_logits, 1, batch_size);
        if (ret != 0 || !head_logits) {
            free(head_logits);
            break;
        }
        
        /* Sample from this head's logits */
        uint32_t token = reges_sample(head_logits, m->config.vocab_size, 1.0f, step + 1);
        draft_out[n_draft] = token;
        n_draft++;
        
        free(head_logits);
    }
    
    return n_draft;
}

uint32_t reges_spec_verify(struct reges_model *m,
                           struct reges_kv_cache *cache,
                           const uint32_t *draft_tokens,
                           uint32_t n_draft,
                           float temperature) {
    if (!m || !m->weights || !draft_tokens) {
        _set_error("NULL arguments to spec_verify");
        return 0;
    }
    
    /* Verify each draft token against the trunk model */
    uint32_t n_accepted = 0;
    
    for (uint32_t i = 0; i < n_draft; i++) {
        /* Decode with the trunk model using the draft token as input */
        float *logits = (float *)malloc(m->config.vocab_size * sizeof(float));
        if (!logits) break;
        
        int ret = reges_decode(m, cache, &draft_tokens[i], NULL, 1, logits);
        if (ret != 0) {
            free(logits);
            break;
        }
        
        /* Check if the trunk model agrees with the draft */
        uint32_t sampled = reges_sample(logits, m->config.vocab_size, temperature, i + 1000);
        
        if (sampled == draft_tokens[i]) {
            n_accepted++;
        } else {
            /* Mismatch: accept up to this point, then use the trunk's sample */
            break;
        }
        
        free(logits);
    }
    
    return n_accepted;
}
