import numpy as np

def save_model(
    path,
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm
):
    parameters = {
        "E": E,
        "P": P,
        "gamma_final": gamma_final,
        "beta_final": beta_final,
        "W_lm": W_lm,
    }

    for layer_idx, block in enumerate(blocks):
        for parameter_name, parameter in block.items():
            parameters[
                f"block{layer_idx}_{parameter_name}"
            ] = parameter

    np.savez(path, **parameters)


def load_model(path):
    data = np.load(path)

    E = data["E"]
    P = data["P"]

    gamma_final = data["gamma_final"]
    beta_final = data["beta_final"]

    W_lm = data["W_lm"]

    blocks = []

    layer_idx = 0

    while f"block{layer_idx}_W_Q1" in data:
        block = {}

        parameter_names = [
            "W_Q1",
            "W_K1",
            "W_V1",
            "W_Q2",
            "W_K2",
            "W_V2",
            "W_O",
            "W1",
            "b1",
            "W2",
            "b2",
            "gamma1",
            "beta1",
            "gamma2",
            "beta2",
        ]

        for parameter_name in parameter_names:
            key = f"block{layer_idx}_{parameter_name}"
            block[parameter_name] = data[key]

        blocks.append(block)

        layer_idx += 1

    return (
        E,
        P,
        blocks,
        gamma_final,
        beta_final,
        W_lm
    )    