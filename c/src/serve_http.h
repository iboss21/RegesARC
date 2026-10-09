#ifndef REGES_SERVE_HTTP_H
#define REGES_SERVE_HTTP_H

/**
 * RegesARC HTTP Server — OpenAI + Anthropic compatible endpoints on one port.
 * 
 * Provides:
 * - POST /v1/chat/completions (OpenAI format)
 * - POST /v1/messages (Anthropic format)
 * - GET /health (status check)
 * - GET /models (list available models)
 */

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Server configuration ───────────────────────────────────────────── */
typedef struct {
    uint16_t port;                /* Port to listen on (default: 8080) */
    const char *host;             /* Bind address (default: "0.0.0.0") */
    bool cors_enabled;            /* Enable CORS headers */
    uint32_t max_batch_size;      /* Maximum concurrent requests */
    uint32_t request_timeout_ms;  /* Request timeout in milliseconds */
} reges_server_config_t;

/* ── Server handle ──────────────────────────────────────────────────── */
typedef struct reges_server {
    pthread_t thread;
    int server_fd;
    reges_server_config_t config;
    
    /* Model registry (populated by engine) */
    char **model_ids;
    uint32_t n_models;
    
    /* Request queue and worker threads */
    pthread_mutex_t queue_lock;
    pthread_cond_t queue_cond;
    struct reges_request *request_queue;
    uint32_t queue_size;
    uint32_t max_queue_size;
} reges_server_t;

/* ── Request structure ──────────────────────────────────────────────── */
typedef enum {
    REGES_REQ_OPENAI = 0,
    REGES_REQ_ANTHROPIC,
} reges_request_type_t;

typedef struct reges_request {
    reges_request_type_t type;
    char *model_id;
    char *prompt;
    uint32_t max_tokens;
    float temperature;
    
    /* Response buffer */
    char *response_json;
    uint32_t response_len;
    
    /* Status */
    bool completed;
    int status_code;
} reges_request_t;

/* ── API ────────────────────────────────────────────────────────────── */

/**
 * Initialize and start the HTTP server.
 * Returns 0 on success, -1 on failure.
 */
int reges_server_start(reges_server_t *server, const reges_server_config_t *config);

/** Stop the HTTP server and free resources. */
void reges_server_stop(reges_server_t *server);

/** Register a model with the server's model registry. */
int reges_server_register_model(reges_server_t *server, const char *model_id);

/** Unregister a model from the registry. */
void reges_server_unregister_model(reges_server_t *server, const char *model_id);

#ifdef __cplusplus
}
#endif

#endif /* REGES_SERVE_HTTP_H */
