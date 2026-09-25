import numpy as np

from .utils import layer_norm, softmax
from .transformer_block_forward import transformer_block_forward


def inference(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    inputs
):
    # Token embeddings
    embeddings = E[inputs]

    # Position embeddings
    positions = np.arange(len(inputs))
    position_embeddings = P[positions]

    # Combine token + position embeddings
    X = embeddings + position_embeddings

    # Transformer blocks
    hidden = X

    for block in blocks:
        hidden, _ = transformer_block_forward(
            hidden,
            block
        )

    # Final LayerNorm
    hidden_norm = layer_norm(
        hidden,
        gamma_final,
        beta_final
    )

    # Language-model head
    logits = hidden_norm @ W_lm

    # Convert logits to probabilities
    probabilities = softmax(
        logits,
        axis=-1
    )

    return logits, probabilities