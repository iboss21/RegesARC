/**
 * RegesARC Public C API — Python layer interface via ctypes/cffi.
 * 
 * This is the stable ABI that the Python control plane (reges/scan.py,
 * reges/catalog.py, reges/place.py, reges/install.py, reges/serve.py) calls
 * into through the compiled engine binary and family adapter shared libraries.
 * 
 * All functions are thread-safe unless otherwise noted.
 */

#ifndef REGES_API_H
#define REGES_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version info (for Python version checking) ─────────────────────── */

/** Get the engine version string. Returns static buffer. */
const char *reges_version(void);

/** Get the build configuration (backend, features). Returns static buffer. */
const char *reges_build_info(void);

/* ── Machine scan (mirrors Python reges/scan.py) ────────────────────── */

typedef struct {
    float ram_gb;
    float gpu_gb;
    float disk_free_gb;
    char store[512];
} reges_machine_info_t;

/** Scan the current machine for resources. Returns 0 on success. */
int reges_scan_machine(const char *store_path, reges_machine_info_t *info);

/* ── Placement verdict (mirrors Python reges/place.py) ──────────────── */

typedef enum {
    REGES_PLACEMENT_REFUSE = 0,
    REGES_PLACEMENT_DISK   = 1,
    REGES_PLACEMENT_CARD   = 2,
} reges_placement_t;

typedef struct {
    char model_id[128];
    reges_placement_t verdict;
    char reason[512];
} reges_placement_result_t;

/**
 * Determine placement for a model given machine resources.
 * model_params: total params in GB, hot_gb: VRAM needed, trunk_gb: RAM needed
 */
int reges_place_model(const char *model_id, float disk_free_gb, 
                      float ram_gb, float gpu_gb,
                      float model_disk_gb, float model_hot_gb, float model_trunk_gb,
                      reges_placement_result_t *result);

/* ── Model loading and inference (mirrors Python reges/serve.py) ─────── */

/** Opaque model handle (returned by reges_load_model). */
typedef void *reges_model_handle_t;

/** Load a model from GGUF file. Returns NULL on failure. */
reges_model_handle_t reges_load_model(const char *gguf_path, 
                                      const char *adapter_path);

/** Free a loaded model and all resources. */
void reges_free_model(reges_model_handle_t handle);

/** Get model info (name, params, family). Returns static buffer JSON. */
const char *reges_model_info(reges_model_handle_t handle);

/**
 * Generate text from a prompt.
 * prompt: input text
 * max_tokens: maximum output tokens
 * temperature: sampling temperature (0 = greedy)
 * output_buf: pre-allocated buffer for output text
 * output_buf_size: size of output buffer
 * Returns number of characters written, or -1 on error.
 */
int reges_generate(reges_model_handle_t handle, const char *prompt,
                   uint32_t max_tokens, float temperature,
                   char *output_buf, uint32_t output_buf_size);

/**
 * Generate text with streaming callback.
 * Each token is passed to the callback as it's generated.
 */
typedef void (*reges_token_callback_t)(uint32_t token_id, const char *text, 
                                       void *user_data);

int reges_generate_stream(reges_model_handle_t handle, const char *prompt,
                          uint32_t max_tokens, float temperature,
                          reges_token_callback_t callback, void *user_data);

/* ── Model catalog (mirrors Python reges/catalog.py) ─────────────────── */

typedef struct {
    char id[128];
    char label[128];
    char params[128];
    float disk_gb;
    float trunk_gb;
    float hot_gb;
    char family[64];
} reges_catalog_entry_t;

/** Get the number of models in the catalog. */
uint32_t reges_catalog_count(void);

/** Get a catalog entry by index. Returns NULL if out of bounds. */
const reges_catalog_entry_t *reges_catalog_get(uint32_t index);

/** Find a catalog entry by ID. Returns NULL if not found. */
const reges_catalog_entry_t *reges_catalog_find(const char *id);

/* ── Family adapter management ──────────────────────────────────────── */

/** List all available family adapters. Returns static buffer JSON array. */
const char *reges_list_families(void);

/** Check if a specific family adapter is available. */
bool reges_family_available(const char *family_name);

/* ── Expert residency management ────────────────────────────────────── */

/** Pin experts to GPU VRAM (hot set). Returns 0 on success. */
int reges_pin_experts(reges_model_handle_t handle, uint32_t n_hot);

/** Stream a specific expert from disk. Returns 0 on success. */
int reges_stream_expert(reges_model_handle_t handle, uint32_t expert_id);

/** Evict an expert back to disk. Returns 0 on success. */
int reges_evict_expert(reges_model_handle_t handle, uint32_t expert_id);

/* ── Speculative decoding control ───────────────────────────────────── */

/** Enable/disable speculative decoding. Returns 0 on success. */
int reges_set_spec_decode(reges_model_handle_t handle, bool enabled);

/** Get spec decode status. Returns true if enabled. */
bool reges_get_spec_decode(reges_model_handle_t handle);

/* ── Performance monitoring ─────────────────────────────────────────── */

typedef struct {
    uint64_t tokens_generated;
    uint64_t tokens_drafted;      /* Speculative decoding drafts */
    uint64_t tokens_accepted;     /* Speculative decoding acceptances */
    double first_token_latency_ms;
    double throughput_tokens_per_sec;
    size_t vram_used_bytes;
    size_t ram_used_bytes;
} reges_perf_stats_t;

/** Get current performance statistics. Returns 0 on success. */
int reges_get_perf_stats(reges_model_handle_t handle, reges_perf_stats_t *stats);

/* ── Error handling ─────────────────────────────────────────────────── */

/** Return the last error message (thread-safe). */
const char *reges_error(void);

#ifdef __cplusplus
}
#endif

#endif /* REGES_API_H */
