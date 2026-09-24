import numpy as np

from .utils import (layer_norm_backward, 
                   ffn_backward, 
                   softmax_backward
                )


def transformer_block_backward(doutput, block, cache):

    d_head = cache["Q1"].shape[-1]

    dR, dgamma2, dbeta2 = layer_norm_backward(
        doutput,
        cache["R"],
        block["gamma2"],
        block["beta2"]
    )

    dZ_from_residual = dR
    dF = dR

    (
        dZ_ffn,
        dW1,
        db1,
        dW2,
        db2
    ) = ffn_backward(
        dF,
        cache["Z"],
        cache["A"],
        cache["H"],
        block["W1"],
        block["W2"]
    )

    dZ = dZ_from_residual + dZ_ffn

    dY, dgamma1, dbeta1 = layer_norm_backward(
        dZ,
        cache["Y"],
        block["gamma1"],
        block["beta1"]
    )

    dX_from_residual = dY
    dattention_output = dY

    dW_O = cache["combined"].T @ dattention_output

    dcombined = dattention_output @ block["W_O"].T

    dO1 = dcombined[:, :d_head]
    dO2 = dcombined[:, d_head:]

    A1 = cache["attention_weights1"]
    V1 = cache["V1"]

    dA1 = dO1 @ V1.T
    dV1 = A1.T @ dO1

    A2 = cache["attention_weights2"]
    V2 = cache["V2"]

    dA2 = dO2 @ V2.T
    dV2 = A2.T @ dO2

    dS1 = softmax_backward(dA1, A1)

    dS2 = softmax_backward(dA2, A2)

    mask = cache["mask"]

    dS1 = np.where(mask == 1, 0.0, dS1)

    dS2 = np.where(mask == 1, 0.0, dS2)

    scale = np.sqrt(d_head)

    dQK1 = dS1 / scale
    dQK2 = dS2 / scale

    Q1 = cache["Q1"]
    K1 = cache["K1"]

    Q2 = cache["Q2"]
    K2 = cache["K2"]

    dQ1 = dQK1 @ K1
    dK1 = dQK1.T @ Q1

    dQ2 = dQK2 @ K2
    dK2 = dQK2.T @ Q2

    X = cache["X"]

    dW_Q1 = X.T @ dQ1
    dW_K1 = X.T @ dK1
    dW_V1 = X.T @ dV1

    dX_Q1 = dQ1 @ block["W_Q1"].T
    dX_K1 = dK1 @ block["W_K1"].T
    dX_V1 = dV1 @ block["W_V1"].T

    dW_Q2 = X.T @ dQ2
    dW_K2 = X.T @ dK2
    dW_V2 = X.T @ dV2

    dX_Q2 = dQ2 @ block["W_Q2"].T
    dX_K2 = dK2 @ block["W_K2"].T
    dX_V2 = dV2 @ block["W_V2"].T

    dX_attention = (
        dX_Q1
        + dX_K1
        + dX_V1
        + dX_Q2
        + dX_K2
        + dX_V2
    )

    dX = (dX_from_residual + dX_attention)

    gradients = {

        "W_Q1": dW_Q1,
        "W_K1": dW_K1,
        "W_V1": dW_V1,

        "W_Q2": dW_Q2,
        "W_K2": dW_K2,
        "W_V2": dW_V2,

        "W_O": dW_O,

        "W1": dW1,
        "b1": db1,

        "W2": dW2,
        "b2": db2,

        "gamma1": dgamma1,
        "beta1": dbeta1,

        "gamma2": dgamma2,
        "beta2": dbeta2
    }

    return dX, gradients