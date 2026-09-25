from .utils import numerical_gradient


def loss_fn():
    loss, _ = forward(
        E,
        P,
        blocks,
        gamma_final,
        beta_final,
        W_lm,
        inputs,
        targets
    )
    return loss

# Numerical gradient
print("\nChecking W_lm:")
numerical_gradient(
    W_lm,
    gradients["W_lm"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking E:")
numerical_gradient(
    E,
    gradients["E"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking P:")
numerical_gradient(
    P,
    gradients["P"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking gamma_final:")
numerical_gradient(
    gamma_final,
    gradients["gamma_final"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking beta_final:")
numerical_gradient(
    beta_final,
    gradients["beta_final"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 0 W_Q1:")
numerical_gradient(
    blocks[0]["W_Q1"],
    gradients["blocks"][0]["W_Q1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W_K1:")
numerical_gradient(
    blocks[0]["W_K1"],
    gradients["blocks"][0]["W_K1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W_V1:")
numerical_gradient(
    blocks[0]["W_V1"],
    gradients["blocks"][0]["W_V1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W_Q2:")
numerical_gradient(
    blocks[0]["W_Q2"],
    gradients["blocks"][0]["W_Q2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W_K2:")
numerical_gradient(
    blocks[0]["W_K2"],
    gradients["blocks"][0]["W_K2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W_V2:")
numerical_gradient(
    blocks[0]["W_V2"],
    gradients["blocks"][0]["W_V2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W_O:")
numerical_gradient(
    blocks[0]["W_O"],
    gradients["blocks"][0]["W_O"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W1:")
numerical_gradient(
    blocks[0]["W1"],
    gradients["blocks"][0]["W1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 b1:")
numerical_gradient(
    blocks[0]["b1"],
    gradients["blocks"][0]["b1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 W2:")
numerical_gradient(
    blocks[0]["W2"],
    gradients["blocks"][0]["W2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 b2:")
numerical_gradient(
    blocks[0]["b2"],
    gradients["blocks"][0]["b2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 0 gamma1:")
numerical_gradient(
    blocks[0]["gamma1"],
    gradients["blocks"][0]["gamma1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 0 beta1:")
numerical_gradient(
    blocks[0]["beta1"],
    gradients["blocks"][0]["beta1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 0 gamma2:")
numerical_gradient(
    blocks[0]["gamma2"],
    gradients["blocks"][0]["gamma2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 0 beta2:")
numerical_gradient(
    blocks[0]["beta2"],
    gradients["blocks"][0]["beta2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\n==============================")
print("BLOCK 1 GRADIENT CHECKS")
print("==============================")

print("\nChecking Block 1 W_Q1:")
numerical_gradient(
    blocks[1]["W_Q1"],
    gradients["blocks"][1]["W_Q1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 1 W_K1:")
numerical_gradient(
    blocks[1]["W_K1"],
    gradients["blocks"][1]["W_K1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 1 W_V1:")
numerical_gradient(
    blocks[1]["W_V1"],
    gradients["blocks"][1]["W_V1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 1 W_O:")
numerical_gradient(
    blocks[1]["W_O"],
    gradients["blocks"][1]["W_O"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 1 W1:")
numerical_gradient(
    blocks[1]["W1"],
    gradients["blocks"][1]["W1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 1 W2:")
numerical_gradient(
    blocks[1]["W2"],
    gradients["blocks"][1]["W2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 1 gamma1:")
numerical_gradient(
    blocks[1]["gamma1"],
    gradients["blocks"][1]["gamma1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 1 beta1:")
numerical_gradient(
    blocks[1]["beta1"],
    gradients["blocks"][1]["beta1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 1 gamma2:")
numerical_gradient(
    blocks[1]["gamma2"],
    gradients["blocks"][1]["gamma2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 1 beta2:")
numerical_gradient(
    blocks[1]["beta2"],
    gradients["blocks"][1]["beta2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\n==============================")
print("BLOCK 2 GRADIENT CHECKS")
print("==============================")

print("\nChecking Block 2 W_Q1:")
numerical_gradient(
    blocks[2]["W_Q1"],
    gradients["blocks"][2]["W_Q1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 2 W_K1:")
numerical_gradient(
    blocks[2]["W_K1"],
    gradients["blocks"][2]["W_K1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 2 W_V1:")
numerical_gradient(
    blocks[2]["W_V1"],
    gradients["blocks"][2]["W_V1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 2 W_O:")
numerical_gradient(
    blocks[2]["W_O"],
    gradients["blocks"][2]["W_O"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 2 W1:")
numerical_gradient(
    blocks[2]["W1"],
    gradients["blocks"][2]["W1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 2 W2:")
numerical_gradient(
    blocks[2]["W2"],
    gradients["blocks"][2]["W2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=5
)

print("\nChecking Block 2 gamma1:")
numerical_gradient(
    blocks[2]["gamma1"],
    gradients["blocks"][2]["gamma1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 2 beta1:")
numerical_gradient(
    blocks[2]["beta1"],
    gradients["blocks"][2]["beta1"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 2 gamma2:")
numerical_gradient(
    blocks[2]["gamma2"],
    gradients["blocks"][2]["gamma2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)

print("\nChecking Block 2 beta2:")
numerical_gradient(
    blocks[2]["beta2"],
    gradients["blocks"][2]["beta2"],
    loss_fn,
    epsilon=1e-5,
    num_checks=4
)