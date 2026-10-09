#ifndef REGES_FAMILY_REGESCORE_H
#define REGES_FAMILY_REGESCORE_H

/**
 * RegesCore 397B Flagship family adapter.
 * 
 * Architecture:
 * - Ultra-large MoE with 397B total parameters
 * - 256 experts with top-8 routing for maximum capacity
 * - MLA (Multi-Latent Attention) for memory efficiency
 * - MTP heads for aggressive speculative decoding
 * - FP8 computation with BF16 accumulation
 * - Expert residency: hot experts pinned to VRAM, cold streamed from disk
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the RegesCore 397B family adapter. */
int reges_family_init(void);

/** Free all adapter resources. */
void reges_family_free(void);

/**
 * Forward pass: MLA attention + ultra-large MoE routing.
 * input: [batch_size][hidden_dim]
 * output: [batch_size][vocab_size] logits
 */
int reges_family_forward(const float *input, float *output, uint32_t batch_size);

/**
 * MTP forward for speculative decoding (aggressive multi-token prediction).
 * outputs: array of [n_heads][batch_size][vocab_size] logits
 */
int reges_family_mtp_forward(const float *input, float **outputs, 
                             uint32_t n_heads, uint32_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* REGES_FAMILY_REGESCORE_H */
