#ifndef REGES_FAMILY_QWEN38_H
#define REGES_FAMILY_QWEN38_H

/**
 * Qwen3.8-Flash-Next family adapter.
 * Implements: GQA attention, SwiGLU MLP, MoE routing with top-k selection,
 * MTP heads for speculative decoding.
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the Qwen3.8 family adapter (allocate working buffers). */
int reges_family_init(void);

/** Free all adapter resources. */
void reges_family_free(void);

/**
 * Forward pass: attention + MLP + MoE routing.
 * input: [batch_size][hidden_dim]
 * output: [batch_size][vocab_size] logits
 */
int reges_family_forward(const float *input, float *output, uint32_t batch_size);

/**
 * MTP forward: generate draft tokens using multi-token prediction heads.
 * outputs: array of [n_heads][batch_size][vocab_size] logits
 */
int reges_family_mtp_forward(const float *input, float **outputs, 
                             uint32_t n_heads, uint32_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* REGES_FAMILY_QWEN38_H */
