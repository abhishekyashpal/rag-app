import numpy as np



def layer_norm(X, gamma, beta, eps=1e-5):
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)

    normalized = (X - mean) / np.sqrt(variance + eps)

    return gamma * normalized + beta

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(probabilities, targets):
    correct_probabilities = probabilities[
        np.arange(len(targets)),
        targets
    ]
    correct_probabilities = np.clip(
        correct_probabilities,
        1e-12,
        1.0
    )
    losses = -np.log(correct_probabilities)
    return np.mean(losses)

def lm_loss(W_lm, hidden_norm, targets):
    logits = hidden_norm @ W_lm

    probabilities = softmax(logits, axis=-1)

    correct_probabilities = probabilities[
        np.arange(len(targets)),
        targets
    ]

    correct_probabilities = np.clip(
        correct_probabilities,
        1e-12,
        1.0
    )
    losses = -np.log(correct_probabilities)
    return np.mean(losses)

def layer_norm_backward(dY, X, gamma, beta, eps=1e-5):

    # Mean and variance
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)

    # Normalized input
    normalized = (X - mean) / np.sqrt(variance + eps)

    # Parameter gradients
    dgamma = np.sum(
        dY * normalized,
        axis=0
    )

    dbeta = np.sum(
        dY,
        axis=0
    )

    # Mean of dY
    mean_dY = np.mean(
        dY,
        axis=-1,
        keepdims=True
    )

    # Mean of dY * normalized
    mean_dY_normalized = np.mean(
        dY * normalized,
        axis=-1,
        keepdims=True
    )

    # Gradient with respect to X
    dX = (
        gamma
        / np.sqrt(variance + eps)
        * (
            dY
            - mean_dY
            - normalized * mean_dY_normalized
        )
    )

    return dX, dgamma, dbeta


def ffn_backward(dF, Z, A, H, W1, W2):
    
    # -------------------------
    # F = H @ W2 + b2
    # -------------------------
    dW2 = H.T @ dF
    db2 = np.sum(dF, axis=0)

    dH = dF @ W2.T

    # -------------------------
    # ReLU
    # -------------------------
    dA = dH * (A > 0)

    # -------------------------
    # A = Z @ W1 + b1
    # -------------------------
    dW1 = Z.T @ dA
    db1 = np.sum(dA, axis=0)

    dZ_ffn = dA @ W1.T

    return (
        dZ_ffn,
        dW1,
        db1,
        dW2,
        db2
    )

def softmax_backward(dA, A):

    sum_dA_A = np.sum(
        dA * A,
        axis=-1,
        keepdims=True
    )

    dS = A * (
        dA - sum_dA_A
    )

    return dS


def numerical_gradient(
    parameter,
    analytical_gradient,
    loss_function,
    epsilon=1e-5,
    num_checks=5
):

    for _ in range(num_checks):

        index = tuple(
            np.random.randint(0, dim)
            for dim in parameter.shape
        )

        original_value = parameter[index]

        parameter[index] = (
            original_value + epsilon
        )

        loss_plus = loss_function()

        parameter[index] = (
            original_value - epsilon
        )

        loss_minus = loss_function()

        parameter[index] = original_value

        numerical = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        analytical = analytical_gradient[index]

        relative_error = (
            abs(numerical - analytical)
            / max(
                1e-8,
                abs(numerical) + abs(analytical)
            )
        )

        print(
            f"{index}: "
            f"numerical={numerical:.8f}, "
            f"analytical={analytical:.8f}, "
            f"error={relative_error:.8e}"
        )