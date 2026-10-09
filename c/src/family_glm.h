#ifndef REGES_FAMILY_GLM_H
#define REGES_FAMILY_GLM_H

/**
 * GLM family adapter (GLM-5.3 Flash / GLM-5.2).
 * 
 * Architecture:
 * - GQA attention with rope scaling
 * - SwiGLU MLP with MoE routing
 * - Specialized tokenization for Chinese/English mixed text
 * - MTP heads for speculative decoding (GLM-5.3 Flash only)
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the GLM family adapter. */
int reges_family_init(void);

/** Free all adapter resources. */
void reges_family_free(void);

/**
 * Forward pass: GQA attention + SwiGLU MLP + MoE routing.
 * input: [batch_size][hidden_dim]
 * output: [batch_size][vocab_size] logits
 */
int reges_family_forward(const float *input, float *output, uint32_t batch_size);

/**
 * MTP forward for speculative decoding (GLM-5.3 Flash).
 * outputs: array of [n_heads][batch_size][vocab_size] logits
 */
int reges_family_mtp_forward(const float *input, float **outputs, 
                             uint32_t n_heads, uint32_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* REGES_FAMILY_GLM_H */
