import numpy as np
from .utils import (layer_norm, 
                   softmax,
                )



def transformer_block_forward(
    X,
    block
):

    # -------------------------
    # Multi-Head Self-Attention
    # -------------------------
    Q1 = X @ block["W_Q1"]
    K1 = X @ block["W_K1"]
    V1 = X @ block["W_V1"]

    Q2 = X @ block["W_Q2"]
    K2 = X @ block["W_K2"]
    V2 = X @ block["W_V2"]

    d_head = Q1.shape[-1]

    scores1 = Q1 @ K1.T / np.sqrt(d_head)
    scores2 = Q2 @ K2.T / np.sqrt(d_head)

    # Causal mask
    T = X.shape[0]

    mask = np.triu(
        np.ones((T, T)),
        k=1
    )

    scores1 = np.where(
        mask == 1,
        -np.inf,
        scores1
    )

    scores2 = np.where(
        mask == 1,
        -np.inf,
        scores2
    )

    # Softmax
    attention_weights1 = softmax(scores1, axis=-1)
    attention_weights2 = softmax(scores2, axis=-1)

    # Attention outputs
    O1 = attention_weights1 @ V1
    O2 = attention_weights2 @ V2

    # Concatenate heads
    combined = np.concatenate(
        [O1, O2],
        axis=-1
    )

    # Output projection
    attention_output = combined @ block["W_O"]

    # -------------------------
    # Residual + LayerNorm
    # -------------------------

    Y = X + attention_output

    Z = layer_norm(
        Y,
        block["gamma1"],
        block["beta1"]
    )

    # -------------------------
    # Feed-Forward Network
    # -------------------------

    A = Z @ block["W1"] + block["b1"]

    H = np.maximum(
        0,
        A
    )

    F = H @ block["W2"] + block["b2"]

    # -------------------------
    # Residual + LayerNorm
    # -------------------------

    R = Z + F

    output = layer_norm(
            R,
            block["gamma2"],
            block["beta2"]
        )

    cache = {
        "X": X,
        "mask": mask,
        "Q1": Q1,
        "K1": K1,
        "V1": V1,

        "Q2": Q2,
        "K2": K2,
        "V2": V2,

        "scores1": scores1,
        "scores2": scores2,

        "attention_weights1": attention_weights1,
        "attention_weights2": attention_weights2,

        "O1": O1,
        "O2": O2,

        "combined": combined,
        "attention_output": attention_output,

        "Y": Y,
        "Z": Z,

        "A": A,
        "H": H,
        "F": F,

        "R": R,
        "output": output
    }

    return output, cache