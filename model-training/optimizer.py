import numpy as np


def sgd_update(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    gradients,
    learning_rate
):

    E -= learning_rate * gradients["E"]
    P -= learning_rate * gradients["P"]

    gamma_final -= learning_rate * gradients["gamma_final"]
    beta_final -= learning_rate * gradients["beta_final"]

    W_lm -= learning_rate * gradients["W_lm"]

    for block, block_grads in zip(
        blocks,
        gradients["blocks"]
    ):

        for parameter_name in block:

            block[parameter_name] -= (
                learning_rate
                * block_grads[parameter_name]
            )