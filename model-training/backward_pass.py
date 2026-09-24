import numpy as np

from .utils import (
    layer_norm_backward,
)

from .transformer_block_backward import transformer_block_backward


def backward(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    inputs,
    targets,
    cache
):

    probabilities = cache["probabilities"]
    hidden_norm = cache["hidden_norm"]
    hidden = cache["hidden"]
    block_caches = cache["block_caches"]

    # =====================================================
    # 1. Cross entropy + softmax backward
    # =====================================================

    dlogits = probabilities.copy()

    dlogits[
        np.arange(len(targets)),
        targets
    ] -= 1

    dlogits /= len(targets)

    # =====================================================
    # 2. LM head
    #
    # logits = hidden_norm @ W_lm
    # =====================================================

    dW_lm = hidden_norm.T @ dlogits

    dhidden_norm = dlogits @ W_lm.T

    # =====================================================
    # 3. Final LayerNorm
    # =====================================================

    dhidden, dgamma_final, dbeta_final = layer_norm_backward(
        dhidden_norm,
        hidden,
        gamma_final,
        beta_final
    )

    # =====================================================
    # 4. Transformer blocks
    # =====================================================

    dX = dhidden

    all_block_grads = [None] * len(blocks)

    for layer_idx in reversed(range(len(blocks))):

        dX, grads = transformer_block_backward(
            dX,
            blocks[layer_idx],
            block_caches[layer_idx]
        )

        all_block_grads[layer_idx] = grads

    # =====================================================
    # 5. Embeddings
    #
    # X = E[inputs] + P[positions]
    # =====================================================

    dE = np.zeros_like(E)

    np.add.at(
        dE,
        inputs,
        dX
    )

    # =====================================================
    # 6. Positional embeddings
    # =====================================================

    positions = cache["positions"]

    dP = np.zeros_like(P)

    np.add.at(
        dP,
        positions,
        dX
    )

    # =====================================================
    # 7. Return all gradients
    # =====================================================

    gradients = {
        "E": dE,
        "P": dP,

        "gamma_final": dgamma_final,
        "beta_final": dbeta_final,

        "W_lm": dW_lm,

        "blocks": all_block_grads
    }

    return gradients