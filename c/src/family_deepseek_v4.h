#ifndef REGES_FAMILY_DEEPSEEK_V4_H
#define REGES_FAMILY_DEEPSEEK_V4_H

/**
 * DeepSeek V4 family adapter (V4 Flash / V4.1 Flash).
 * 
 * Architecture:
 * - MLA (Multi-Latent Attention) with single KV head per group
 * - Shared experts across all layers (reduced MoE overhead)
 * - Hybrid routing: shared + routed experts
 * - FP8 computation with BF16 accumulation
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the DeepSeek V4 family adapter. */
int reges_family_init(void);

/** Free all adapter resources. */
void reges_family_free(void);

/**
 * Forward pass: MLA attention + hybrid MoE (shared + routed experts).
 * input: [batch_size][hidden_dim]
 * output: [batch_size][vocab_size] logits
 */
int reges_family_forward(const float *input, float *output, uint32_t batch_size);

/**
 * MTP forward for speculative decoding.
 * outputs: array of [n_heads][batch_size][vocab_size] logits
 */
int reges_family_mtp_forward(const float *input, float **outputs, 
                             uint32_t n_heads, uint32_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* REGES_FAMILY_DEEPSEEK_V4_H */
