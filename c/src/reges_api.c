/**
 * RegesARC Public C API Implementation — Python layer interface via ctypes/cffi.
 * 
 * This file implements the stable ABI defined in reges_api.h.
 * The Python control plane calls these functions through ctypes.
 */

#include "reges_api.h"
#include "../src/engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Internal model wrapper (opaque to Python) ──────────────────────── */
typedef struct {
    struct reges_model *engine_model;
    char family[64];
    bool spec_decode_enabled;
    reges_perf_stats_t perf_stats;
} reges_model_wrapper_t;

/* ── Thread-local error buffer ──────────────────────────────────────── */
static __thread char _reges_api_errbuf[512];

const char *reges_error(void) {
    return _reges_api_errbuf;
}

static void _set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(_reges_api_errbuf, sizeof(_reges_api_errbuf), fmt, ap);
    va_end(ap);
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  VERSION AND BUILD INFO                                                  */
/* ════════════════════════════════════════════════════════════════════════ */

const char *reges_version(void) {
    return "1.0.0";
}

const char *reges_build_info(void) {
#ifdef REGES_BACKEND_CUDA
    return "CUDA backend, FP8 support, MTP spec decode";
#elif defined(REGES_BACKEND_HIP)
    return "HIP/ROCm backend, FP8 support, MTP spec decode";
#else
    return "CPU fallback backend, FP32 computation";
#endif
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  MACHINE SCAN (mirrors Python reges/scan.py)                             */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_scan_machine(const char *store_path, reges_machine_info_t *info) {
    if (!info) return -1;
    
    /* In a real implementation, this would call system APIs to get:
     * - Total RAM (Windows: GlobalMemoryStatusEx, Linux: /proc/meminfo)
     * - GPU memory (nvidia-smi or ROCm-smi)
     * - Disk free space (shutil.disk_usage or statvfs)
     */
    
    /* For now, return placeholder values */
    info->ram_gb = 64.0f;
    info->gpu_gb = 24.0f;
    info->disk_free_gb = 500.0f;
    
    if (store_path) {
        strncpy(info->store, store_path, sizeof(info->store) - 1);
    } else {
        strcpy(info->store, ".");
    }
    
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  PLACEMENT VERDICT (mirrors Python reges/place.py)                        */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_place_model(const char *model_id, float disk_free_gb, 
                      float ram_gb, float gpu_gb,
                      float model_disk_gb, float model_hot_gb, float model_trunk_gb,
                      reges_placement_result_t *result) {
    if (!result || !model_id) return -1;
    
    strncpy(result->model_id, model_id, sizeof(result->model_id) - 1);
    
    /* Refuse if not enough disk space */
    if (disk_free_gb < model_disk_gb) {
        result->verdict = REGES_PLACEMENT_REFUSE;
        snprintf(result->reason, sizeof(result->reason),
                "need ~%.0f GB free disk, have %.1f GB", 
                model_disk_gb, disk_free_gb);
        return 0;
    }
    
    /* Card path: hot set fits in GPU VRAM */
    if (gpu_gb >= model_hot_gb && ram_gb >= model_trunk_gb) {
        result->verdict = REGES_PLACEMENT_CARD;
        snprintf(result->reason, sizeof(result->reason),
                "hot set %.0f GB fits in %.1f GB GPU", 
                model_hot_gb, gpu_gb);
        return 0;
    }
    
    /* Disk path: trunk fits in RAM, experts stream from disk */
    if (ram_gb >= model_trunk_gb) {
        result->verdict = REGES_PLACEMENT_DISK;
        snprintf(result->reason, sizeof(result->reason),
                "trunk %.0f GB fits in %.1f GB RAM; experts stream from disk", 
                model_trunk_gb, ram_gb);
        return 0;
    }
    
    /* Refuse: not enough RAM for trunk */
    result->verdict = REGES_PLACEMENT_REFUSE;
    snprintf(result->reason, sizeof(result->reason),
            "trunk needs %.0f GB RAM, have %.1f GB", 
            model_trunk_gb, ram_gb);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  MODEL LOADING AND INFERENCE                                             */
/* ════════════════════════════════════════════════════════════════════════ */

reges_model_handle_t reges_load_model(const char *gguf_path, 
                                      const char *adapter_path) {
    if (!gguf_path || !adapter_path) {
        _set_error("NULL arguments to reges_load_model");
        return NULL;
    }
    
    /* Load the engine model */
    struct reges_model *engine_model = reges_model_load(gguf_path, adapter_path);
    if (!engine_model) {
        _set_error("Failed to load model: %s", reges_errmsg());
        return NULL;
    }
    
    /* Create wrapper */
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)calloc(1, sizeof(reges_model_wrapper_t));
    if (!wrapper) {
        reges_model_free(engine_model);
        _set_error("Out of memory allocating model wrapper");
        return NULL;
    }
    
    wrapper->engine_model = engine_model;
    const reges_model_config_t *config = reges_model_config(engine_model);
    strncpy(wrapper->family, config->family, sizeof(wrapper->family) - 1);
    wrapper->spec_decode_enabled = false;
    
    /* Initialize perf stats */
    memset(&wrapper->perf_stats, 0, sizeof(reges_perf_stats_t));
    
    return (reges_model_handle_t)wrapper;
}

void reges_free_model(reges_model_handle_t handle) {
    if (!handle) return;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    
    if (wrapper->engine_model) {
        reges_model_free(wrapper->engine_model);
    }
    
    free(wrapper);
}

const char *reges_model_info(reges_model_handle_t handle) {
    static char json_buf[1024];
    
    if (!handle) return NULL;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    const reges_model_config_t *config = reges_model_config(wrapper->engine_model);
    
    snprintf(json_buf, sizeof(json_buf),
            "{\"family\":\"%s\",\"layers\":%u,\"heads\":%u,"
            "\"hidden_dim\":%u,\"n_experts\":%u,\"vocab_size\":%u}",
            config->family, config->n_layers, config->n_heads,
            config->hidden_dim, config->n_experts, config->vocab_size);
    
    return json_buf;
}

int reges_generate(reges_model_handle_t handle, const char *prompt,
                   uint32_t max_tokens, float temperature,
                   char *output_buf, uint32_t output_buf_size) {
    if (!handle || !prompt || !output_buf) return -1;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    
    /* Simplified generation: echo back with "Generated:" prefix */
    uint32_t prompt_len = (uint32_t)strlen(prompt);
    uint32_t max_output = output_buf_size - 20; /* Reserve space for prefix */
    
    if (max_tokens < max_output) {
        max_output = max_tokens;
    }
    
    snprintf(output_buf, output_buf_size, "Generated response to: %.*s", 
             (int)max_output, prompt);
    
    /* Update perf stats */
    wrapper->perf_stats.tokens_generated += max_output;
    wrapper->perf_stats.throughput_tokens_per_sec = 50.0f; /* Placeholder */
    
    return (int)strlen(output_buf);
}

int reges_generate_stream(reges_model_handle_t handle, const char *prompt,
                          uint32_t max_tokens, float temperature,
                          reges_token_callback_t callback, void *user_data) {
    if (!handle || !prompt || !callback) return -1;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    
    /* Simplified streaming: call callback for each "token" (word) */
    const char *words[100];
    uint32_t n_words = 0;
    
    /* Split prompt into words (simplified) */
    char *prompt_copy = strdup(prompt);
    if (!prompt_copy) return -1;
    
    char *token = strtok(prompt_copy, " ");
    while (token && n_words < 100) {
        words[n_words++] = token;
        token = strtok(NULL, " ");
    }
    
    /* Call callback for each word */
    for (uint32_t i = 0; i < n_words && i < max_tokens; i++) {
        callback(i, words[i], user_data);
    }
    
    free(prompt_copy);
    
    return (int)n_words;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  MODEL CATALOG (mirrors Python reges/catalog.py)                         */
/* ════════════════════════════════════════════════════════════════════════ */

static const reges_catalog_entry_t _catalog[] = {
    {"regescore-1.0-35", "RegesCore 1.0 35B", "397B MoE — Flagship", 69, 16, 24, "regescore"},
    {"qwen3.8-flash-next", "Qwen3.8-Flash-Next", "125B MoE", 80, 16, 12, "qwen38"},
    {"qwen3.6", "Qwen3.6-35B-A3B", "35B MoE", 22, 8, 8, "qwen36"},
    {"deepseek-v4-flash", "DeepSeek V4 Flash", "284B MoE", 160, 16, 24, "deepseek-v4"},
    {"deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "552B MoE", 510, 24, 48, "deepseek-v4"},
    {"glm-5.3-flash", "GLM-5.3-Flash", "321B MoE", 180, 16, 24, "glm"},
    {"glm-5.2", "GLM-5.2", "744B MoE", 372, 12, 48, "glm"},
    {"inkling", "Inkling", "975B MoE", 500, 24, 48, "inkling"},
    {"kimi-k3", "Kimi K3", "2.8T MoE", 1400, 32, 80, "kimi"},
    {"olmoe", "OLMoE-7B-7B", "7B MoE", 6, 4, 6, "olmoe"},
    {"laya", "Laya", "MoE", 20, 8, 10, "laya"},
};

uint32_t reges_catalog_count(void) {
    return sizeof(_catalog) / sizeof(_catalog[0]);
}

const reges_catalog_entry_t *reges_catalog_get(uint32_t index) {
    if (index >= reges_catalog_count()) return NULL;
    return &_catalog[index];
}

const reges_catalog_entry_t *reges_catalog_find(const char *id) {
    for (uint32_t i = 0; i < reges_catalog_count(); i++) {
        if (strcmp(_catalog[i].id, id) == 0) {
            return &_catalog[i];
        }
    }
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  FAMILY ADAPTER MANAGEMENT                                               */
/* ════════════════════════════════════════════════════════════════════════ */

const char *reges_list_families(void) {
    static char json_buf[256];
    snprintf(json_buf, sizeof(json_buf), 
            "[\"qwen38\",\"deepseek-v4\",\"glm\",\"regescore\"]");
    return json_buf;
}

bool reges_family_available(const char *family_name) {
    if (!family_name) return false;
    
    /* Check against known families */
    const char *families[] = {"qwen38", "deepseek-v4", "glm", "regescore"};
    for (uint32_t i = 0; i < 4; i++) {
        if (strcmp(families[i], family_name) == 0) {
            return true;
        }
    }
    
    return false;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  EXPERT RESIDENCY MANAGEMENT                                               */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_pin_experts(reges_model_handle_t handle, uint32_t n_hot) {
    if (!handle) return -1;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    return reges_expert_pin(wrapper->engine_model, n_hot);
}

int reges_stream_expert(reges_model_handle_t handle, uint32_t expert_id) {
    if (!handle) return -1;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    return reges_expert_stream_in(wrapper->engine_model, expert_id);
}

int reges_evict_expert(reges_model_handle_t handle, uint32_t expert_id) {
    if (!handle) return -1;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    return reges_expert_evict(wrapper->engine_model, expert_id);
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  SPECULATIVE DECODING CONTROL                                              */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_set_spec_decode(reges_model_handle_t handle, bool enabled) {
    if (!handle) return -1;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    wrapper->spec_decode_enabled = enabled;
    return 0;
}

bool reges_get_spec_decode(reges_model_handle_t handle) {
    if (!handle) return false;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    return wrapper->spec_decode_enabled;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  PERFORMANCE MONITORING                                                    */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_get_perf_stats(reges_model_handle_t handle, reges_perf_stats_t *stats) {
    if (!handle || !stats) return -1;
    
    reges_model_wrapper_t *wrapper = (reges_model_wrapper_t *)handle;
    memcpy(stats, &wrapper->perf_stats, sizeof(reges_perf_stats_t));
    
    return 0;
}
