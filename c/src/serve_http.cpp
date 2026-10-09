/**
 * RegesARC HTTP Server — OpenAI + Anthropic compatible endpoints on one port.
 * 
 * Provides:
 * - POST /v1/chat/completions (OpenAI format)
 * - POST /v1/messages (Anthropic format)
 * - GET /health (status check)
 * - GET /models (list available models)
 */

#include "serve_http.h"
#include "../engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define REGES_SOCKET int
#define REGES_CLOSE_SOCKET closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#define REGES_SOCKET int
#define REGES_CLOSE_SOCKET close
#endif

/* ── Request handler function type ─────────────────────────────────── */
typedef void (*reges_request_handler_t)(reges_server_t *server, 
                                        reges_request_t *req);

/* ════════════════════════════════════════════════════════════════════════ */
/*  HTTP PARSING HELPERS                                                    */
/* ════════════════════════════════════════════════════════════════════════ */

static char *_read_request_body(REGES_SOCKET client_fd, uint32_t *len_out) {
    /* Read Content-Length header to determine body size */
    char header_buf[4096];
    int bytes_read = recv(client_fd, header_buf, sizeof(header_buf) - 1, 0);
    if (bytes_read <= 0) return NULL;
    
    header_buf[bytes_read] = '\0';
    
    /* Parse Content-Length */
    uint32_t content_length = 0;
    char *cl_header = strstr(header_buf, "Content-Length:");
    if (cl_header) {
        content_length = (uint32_t)atoi(cl_header + strlen("Content-Length:"));
    }
    
    /* Read the body */
    char *body = (char *)malloc(content_length + 1);
    if (!body) return NULL;
    
    uint32_t total_read = 0;
    while (total_read < content_length) {
        int n = recv(client_fd, body + total_read, content_length - total_read, 0);
        if (n <= 0) break;
        total_read += (uint32_t)n;
    }
    
    body[total_read] = '\0';
    *len_out = total_read;
    return body;
}

static void _send_response(REGES_SOCKET client_fd, int status_code, 
                           const char *content_type, const char *body) {
    char response[8192];
    int len = snprintf(response, sizeof(response),
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: %s\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Connection: close\r\n"
                      "\r\n"
                      "%s",
                      status_code,
                      status_code == 200 ? "OK" : "Error",
                      content_type,
                      body);
    
    send(client_fd, response, len, 0);
}

static void _send_json_response(REGES_SOCKET client_fd, int status_code, 
                                const char *json_body) {
    _send_response(client_fd, status_code, "application/json", json_body);
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  REQUEST HANDLERS                                                        */
/* ════════════════════════════════════════════════════════════════════════ */

static void _handle_health(reges_server_t *server, REGES_SOCKET client_fd) {
    char json[1024];
    snprintf(json, sizeof(json),
            "{\"status\":\"ok\",\"version\":\"1.0.0\","
            "\"supported_models\":%u,\"endpoints\":[\"/v1/chat/completions\",\"/v1/messages\"]}",
            server->n_models);
    
    _send_json_response(client_fd, 200, json);
}

static void _handle_models(reges_server_t *server, REGES_SOCKET client_fd) {
    char json[4096];
    int len = snprintf(json, sizeof(json), "{\"models\":[");
    
    for (uint32_t i = 0; i < server->n_models; i++) {
        if (i > 0) len += snprintf(json + len, sizeof(json) - len, ",");
        len += snprintf(json + len, sizeof(json) - len, "\"%s\"", server->model_ids[i]);
    }
    
    len += snprintf(json + len, sizeof(json) - len, "]}");
    
    _send_json_response(client_fd, 200, json);
}

static void _handle_openai_completion(reges_server_t *server, REGES_SOCKET client_fd) {
    uint32_t body_len = 0;
    char *body = _read_request_body(client_fd, &body_len);
    if (!body) {
        _send_json_response(client_fd, 400, "{\"error\":\"Invalid request\"}");
        return;
    }
    
    /* Parse JSON to extract model_id and prompt (simplified parsing) */
    char *model_id = NULL;
    char *prompt = NULL;
    
    /* Find "model" field */
    char *model_ptr = strstr(body, "\"model\":");
    if (model_ptr) {
        model_ptr += strlen("\"model\":");
        while (*model_ptr == ' ') model_ptr++;
        if (*model_ptr == '"') {
            model_ptr++;
            char *end = strchr(model_ptr, '"');
            if (end) {
                uint32_t len = (uint32_t)(end - model_ptr);
                model_id = (char *)malloc(len + 1);
                memcpy(model_id, model_ptr, len);
                model_id[len] = '\0';
            }
        }
    }
    
    /* Find "messages" array and extract user message */
    char *messages_ptr = strstr(body, "\"messages\":");
    if (messages_ptr) {
        messages_ptr += strlen("\"messages\":");
        /* Look for the last user message (simplified) */
        char *user_ptr = strstr(messages_ptr, "\"role\":\"user\"");
        if (user_ptr) {
            user_ptr = strstr(user_ptr, "\"content\":");
            if (user_ptr) {
                user_ptr += strlen("\"content\":");
                while (*user_ptr == ' ') user_ptr++;
                if (*user_ptr == '"') {
                    user_ptr++;
                    char *end = strchr(user_ptr, '"');
                    if (end) {
                        uint32_t len = (uint32_t)(end - user_ptr);
                        prompt = (char *)malloc(len + 1);
                        memcpy(prompt, user_ptr, len);
                        prompt[len] = '\0';
                    }
                }
            }
        }
    }
    
    if (!model_id || !prompt) {
        free(body);
        free(model_id);
        free(prompt);
        _send_json_response(client_fd, 400, "{\"error\":\"Missing model or prompt\"}");
        return;
    }
    
    /* Check if model is registered */
    bool found = false;
    for (uint32_t i = 0; i < server->n_models; i++) {
        if (strcmp(server->model_ids[i], model_id) == 0) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        free(body);
        free(model_id);
        free(prompt);
        _send_json_response(client_fd, 404, 
                           "{\"error\":\"Model not found\",\"supported\":true}");
        return;
    }
    
    /* Generate response (simplified: echo back with "Generated:" prefix) */
    char response[8192];
    snprintf(response, sizeof(response),
            "{\"id\":\"chatcmpl-%s\","
            "\"object\":\"chat.completion\","
            "\"created\":%lu,"
            "\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":\"Generated response to: %s\"},"
            "\"finish_reason\":\"stop\"}],"
            "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}",
            model_id, (unsigned long)time(NULL), model_id, prompt);
    
    _send_json_response(client_fd, 200, response);
    
    free(body);
    free(model_id);
    free(prompt);
}

static void _handle_anthropic_message(reges_server_t *server, REGES_SOCKET client_fd) {
    uint32_t body_len = 0;
    char *body = _read_request_body(client_fd, &body_len);
    if (!body) {
        _send_json_response(client_fd, 400, "{\"error\":\"Invalid request\"}");
        return;
    }
    
    /* Parse JSON to extract model_id and prompt (simplified parsing) */
    char *model_id = NULL;
    char *prompt = NULL;
    
    /* Find "model" field */
    char *model_ptr = strstr(body, "\"model\":");
    if (model_ptr) {
        model_ptr += strlen("\"model\":");
        while (*model_ptr == ' ') model_ptr++;
        if (*model_ptr == '"') {
            model_ptr++;
            char *end = strchr(model_ptr, '"');
            if (end) {
                uint32_t len = (uint32_t)(end - model_ptr);
                model_id = (char *)malloc(len + 1);
                memcpy(model_id, model_ptr, len);
                model_id[len] = '\0';
            }
        }
    }
    
    /* Find "messages" array and extract user message */
    char *messages_ptr = strstr(body, "\"messages\":");
    if (messages_ptr) {
        messages_ptr += strlen("\"messages\":");
        /* Look for the last user message (simplified) */
        char *user_ptr = strstr(messages_ptr, "\"role\":\"user\"");
        if (user_ptr) {
            user_ptr = strstr(user_ptr, "\"content\":");
            if (user_ptr) {
                user_ptr += strlen("\"content\":");
                while (*user_ptr == ' ') user_ptr++;
                if (*user_ptr == '"') {
                    user_ptr++;
                    char *end = strchr(user_ptr, '"');
                    if (end) {
                        uint32_t len = (uint32_t)(end - user_ptr);
                        prompt = (char *)malloc(len + 1);
                        memcpy(prompt, user_ptr, len);
                        prompt[len] = '\0';
                    }
                }
            }
        }
    }
    
    if (!model_id || !prompt) {
        free(body);
        free(model_id);
        free(prompt);
        _send_json_response(client_fd, 400, "{\"error\":\"Missing model or prompt\"}");
        return;
    }
    
    /* Check if model is registered */
    bool found = false;
    for (uint32_t i = 0; i < server->n_models; i++) {
        if (strcmp(server->model_ids[i], model_id) == 0) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        free(body);
        free(model_id);
        free(prompt);
        _send_json_response(client_fd, 404, 
                           "{\"error\":\"Model not found\",\"supported\":true}");
        return;
    }
    
    /* Generate response (simplified: echo back with "Generated:" prefix) */
    char response[8192];
    snprintf(response, sizeof(response),
            "{\"id\":\"msg_%s\","
            "\"type\":\"message\","
            "\"role\":\"assistant\","
            "\"content\":[{\"type\":\"text\",\"text\":\"Generated response to: %s\"}],"
            "\"stop_reason\":\"end_turn\","
            "\"model\":\"%s\"}",
            model_id, prompt, model_id);
    
    _send_json_response(client_fd, 200, response);
    
    free(body);
    free(model_id);
    free(prompt);
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  SERVER THREAD                                                           */
/* ════════════════════════════════════════════════════════════════════════ */

static void *_server_thread(void *arg) {
    reges_server_t *server = (reges_server_t *)arg;
    
    /* Create server socket */
    REGES_SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "[serve] Failed to create socket\n");
        return NULL;
    }
    
    /* Set socket options */
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    /* Bind to address and port */
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(server->config.host ? server->config.host : "0.0.0.0");
    addr.sin_port = htons(server->config.port);
    
    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[serve] Failed to bind to %s:%u\n", 
                server->config.host ? server->config.host : "0.0.0.0", 
                server->config.port);
        REGES_CLOSE_SOCKET(server_fd);
        return NULL;
    }
    
    /* Listen for connections */
    if (listen(server_fd, 128) < 0) {
        fprintf(stderr, "[serve] Failed to listen\n");
        REGES_CLOSE_SOCKET(server_fd);
        return NULL;
    }
    
    server->server_fd = server_fd;
    
    fprintf(stderr, "[serve] HTTP server listening on %s:%u\n", 
            server->config.host ? server->config.host : "0.0.0.0", 
            server->config.port);
    
    /* Accept and handle connections */
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        REGES_SOCKET client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            continue;
        }
        
        /* Read request line */
        char request_line[1024];
        int bytes_read = recv(client_fd, request_line, sizeof(request_line) - 1, 0);
        if (bytes_read <= 0) {
            REGES_CLOSE_SOCKET(client_fd);
            continue;
        }
        
        request_line[bytes_read] = '\0';
        
        /* Parse method and path */
        char method[16];
        char path[256];
        sscanf(request_line, "%s %s", method, path);
        
        /* Route to appropriate handler */
        if (strcmp(method, "GET") == 0) {
            if (strcmp(path, "/health") == 0 || strcmp(path, "/") == 0) {
                _handle_health(server, client_fd);
            } else if (strcmp(path, "/models") == 0) {
                _handle_models(server, client_fd);
            } else {
                _send_json_response(client_fd, 404, "{\"error\":\"Not found\"}");
            }
        } else if (strcmp(method, "POST") == 0) {
            if (strcmp(path, "/v1/chat/completions") == 0) {
                _handle_openai_completion(server, client_fd);
            } else if (strcmp(path, "/v1/messages") == 0) {
                _handle_anthropic_message(server, client_fd);
            } else {
                _send_json_response(client_fd, 404, "{\"error\":\"Not found\"}");
            }
        } else {
            _send_json_response(client_fd, 405, "{\"error\":\"Method not allowed\"}");
        }
        
        REGES_CLOSE_SOCKET(client_fd);
    }
    
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════ */
/*  PUBLIC API                                                              */
/* ════════════════════════════════════════════════════════════════════════ */

int reges_server_start(reges_server_t *server, const reges_server_config_t *config) {
    if (!server || !config) return -1;
    
    memcpy(&server->config, config, sizeof(reges_server_config_t));
    
    /* Initialize model registry */
    server->model_ids = NULL;
    server->n_models = 0;
    
    /* Initialize request queue */
    pthread_mutex_init(&server->queue_lock, NULL);
    pthread_cond_init(&server->queue_cond, NULL);
    server->request_queue = NULL;
    server->queue_size = 0;
    server->max_queue_size = config->max_batch_size * 10;
    
    /* Start server thread */
    if (pthread_create(&server->thread, NULL, _server_thread, server) != 0) {
        fprintf(stderr, "[serve] Failed to create server thread\n");
        return -1;
    }
    
    return 0;
}

void reges_server_stop(reges_server_t *server) {
    if (!server) return;
    
    /* Close server socket (this will cause accept() to fail and thread to exit) */
    if (server->server_fd >= 0) {
        REGES_CLOSE_SOCKET(server->server_fd);
    }
    
    /* Wait for thread to exit */
    pthread_join(server->thread, NULL);
    
    /* Free model registry */
    if (server->model_ids) {
        for (uint32_t i = 0; i < server->n_models; i++) {
            free(server->model_ids[i]);
        }
        free(server->model_ids);
    }
    
    /* Cleanup request queue */
    reges_request_t *req = server->request_queue;
    while (req) {
        reges_request_t *next = req->next;
        free(req->model_id);
        free(req->prompt);
        free(req->response_json);
        free(req);
        req = next;
    }
    
    pthread_mutex_destroy(&server->queue_lock);
    pthread_cond_destroy(&server->queue_cond);
}

int reges_server_register_model(reges_server_t *server, const char *model_id) {
    if (!server || !model_id) return -1;
    
    /* Check if already registered */
    for (uint32_t i = 0; i < server->n_models; i++) {
        if (strcmp(server->model_ids[i], model_id) == 0) {
            return 0;  /* Already registered */
        }
    }
    
    /* Add to registry */
    char *id_copy = strdup(model_id);
    if (!id_copy) return -1;
    
    server->model_ids = (char **)realloc(server->model_ids, 
                                         (server->n_models + 1) * sizeof(char *));
    if (!server->model_ids) {
        free(id_copy);
        return -1;
    }
    
    server->model_ids[server->n_models] = id_copy;
    server->n_models++;
    
    return 0;
}

void reges_server_unregister_model(reges_server_t *server, const char *model_id) {
    if (!server || !model_id) return;
    
    for (uint32_t i = 0; i < server->n_models; i++) {
        if (strcmp(server->model_ids[i], model_id) == 0) {
            free(server->model_ids[i]);
            
            /* Shift remaining models */
            for (uint32_t j = i; j < server->n_models - 1; j++) {
                server->model_ids[j] = server->model_ids[j + 1];
            }
            
            server->n_models--;
            break;
        }
    }
}
