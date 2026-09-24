import numpy as np

from .utils import (
    layer_norm,
    softmax,
    cross_entropy_loss
)

from .transformer_block_forward import transformer_block_forward


def forward(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    inputs,
    targets
):

    # =====================================================
    # Embeddings
    # =====================================================

    embeddings = E[inputs]

    positions = np.arange(len(inputs))

    position_embeddings = P[positions]

    X = embeddings + position_embeddings

    # =====================================================
    # Transformer blocks
    # =====================================================

    hidden = X

    caches = []

    for block in blocks:

        hidden, cache = transformer_block_forward(
            hidden,
            block
        )

        caches.append(cache)

    # =====================================================
    # Final LayerNorm
    # =====================================================

    hidden_norm = layer_norm(
        hidden,
        gamma_final,
        beta_final
    )

    # =====================================================
    # Language model head
    # =====================================================

    logits = hidden_norm @ W_lm

    probabilities = softmax(
        logits,
        axis=-1
    )

    loss = cross_entropy_loss(
        probabilities,
        targets
    )

    cache = {
        "embeddings": embeddings,
        "positions": positions,
        "position_embeddings": position_embeddings,
        "X": X,
        "hidden": hidden,
        "hidden_norm": hidden_norm,
        "logits": logits,
        "probabilities": probabilities,
        "targets": targets,
        "block_caches": caches
    }

    return loss, cache