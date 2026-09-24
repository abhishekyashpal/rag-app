import numpy as np
np.random.seed(42)
from .forward_pass import forward
from .backward_pass import backward
from .utils import numerical_gradient
from .optimizer import sgd_update

def xavier_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(
        2.0 / (fan_in + fan_out)
    )


text = "the cat sat on the mat"

tokens = text.split(" ")
print(tokens)

vocab = sorted(set(tokens))
print(vocab)

token_to_id = {}

for idx, token in enumerate(vocab):
    token_to_id[token] = idx

print(token_to_id)

id_to_token = {}

for token, idx in token_to_id.items():
    id_to_token[idx] = token

print(id_to_token)

token_ids = np.array(
                [token_to_id[token] for token in tokens],
                dtype=np.int64
            )
print('token_ids:', token_ids)
inputs  = token_ids[:-1]
targets = token_ids[1:]
print("inputs:", inputs)
print("targets:", targets)


vocab_size = len(vocab)
embedding_dim = 4
d_model = embedding_dim

num_heads = 2
assert d_model % num_heads == 0

d_head = d_model // num_heads

d_ff = 16

num_layers = 3

max_context_length = 10

E = np.random.randn(
    vocab_size,
    embedding_dim
) * 0.02
print('Embedding matrix', E.shape)
P = np.random.randn(
    max_context_length,
    embedding_dim
) * 0.02


###################Transformer Block Initialization####################

blocks = []
for _ in range(num_layers):
    block = {
        "W_Q1": xavier_init(d_model, d_head),
        "W_K1": xavier_init(d_model, d_head),
        "W_V1": xavier_init(d_model, d_head),
        "W_Q2": xavier_init(d_model, d_head),
        "W_K2": xavier_init(d_model, d_head),
        "W_V2": xavier_init(d_model, d_head),
        "W_O": xavier_init(d_model, d_model),
        "gamma1": np.ones(d_model),
        "beta1": np.zeros(d_model),
        "W1": xavier_init(d_model, d_ff),
        "b1": np.zeros(d_ff),
        "W2": xavier_init(d_ff, d_model),
        "b2": np.zeros(d_model),
        "gamma2": np.ones(d_model),
        "beta2": np.zeros(d_model)
    }
    blocks.append(block)



########## Final LayerNorm
gamma_final = np.ones(d_model)
beta_final = np.zeros(d_model)


############# Language-model head
W_lm = xavier_init(d_model, vocab_size)


loss, cache = forward(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    inputs,
    targets
)

print("Loss:", loss)


gradients = backward(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    inputs,
    targets,
    cache
)

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

print("dE shape:", gradients["E"].shape)
print("dP shape:", gradients["P"].shape)
print("dgamma_final shape:", gradients["gamma_final"].shape)
print("dbeta_final shape:", gradients["beta_final"].shape)
print("dW_lm shape:", gradients["W_lm"].shape)

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

# print("\n==============================")
# print("ONE-STEP SGD TEST")
# print("==============================")

# loss_before, _ = forward(
#     E,
#     P,
#     blocks,
#     gamma_final,
#     beta_final,
#     W_lm,
#     inputs,
#     targets
# )

# print("Loss before update:", loss_before)

# learning_rate = 0.0001

# sgd_update(
#     E,
#     P,
#     blocks,
#     gamma_final,
#     beta_final,
#     W_lm,
#     gradients,
#     learning_rate
# )

# loss_after, _ = forward(
#     E,
#     P,
#     blocks,
#     gamma_final,
#     beta_final,
#     W_lm,
#     inputs,
#     targets
# )

# print("Loss after update:", loss_after)
# print("Loss change:", loss_after - loss_before)


print()
print("==============================")
print("TRAINING")
print("==============================")

learning_rate = 0.0001
num_steps = 1000

for step in range(num_steps):

    # Forward pass
    loss, cache = forward(
        E,
        P,
        blocks,
        gamma_final,
        beta_final,
        W_lm,
        inputs,
        targets
    )

    # Backward pass
    gradients = backward(
        E,
        P,
        blocks,
        gamma_final,
        beta_final,
        W_lm,
        inputs,
        targets,
        cache
    )

    # SGD update
    sgd_update(
        E,
        P,
        blocks,
        gamma_final,
        beta_final,
        W_lm,
        gradients,
        learning_rate
    )

    # Print progress
    if step % 100 == 0:
        print(f"Step {step:4d} | Loss: {loss:.6f}")
