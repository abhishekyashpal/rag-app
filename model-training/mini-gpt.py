import numpy as np
np.random.seed(42)
from .forward_pass import forward
from .backward_pass import backward
from .optimizer import sgd_update
from .inference import inference
from .generate import generate
from.save_model import save_model, load_model

def xavier_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(
        2.0 / (fan_in + fan_out)
    )


text = "the cat sat on the mat"

tokens = text.split()
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


print()
print("==============================")
print("TRAINING")
print("==============================")

learning_rate = 0.0001
num_steps = 10000

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
    if step % 1000 == 0:
        print(f"Step {step:4d} | Loss: {loss:.6f}")

# Prediction BEFORE saving
test_inputs = np.array(
    [token_to_id["the"]],
    dtype=np.int64
)

logits_before, probabilities_before = inference(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    test_inputs
)

save_model(
    "mini_gpt.npz",
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm
)        

(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm
) = load_model("mini_gpt.npz")

# Prediction AFTER loading
logits_after, probabilities_after = inference(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    test_inputs
)


print(
    "Logits identical:",
    np.allclose(logits_before, logits_after)
)

print(
    "Probabilities identical:",
    np.allclose(
        probabilities_before,
        probabilities_after
    )
)

print()
print("==============================")
print("INFERENCE TEST")
print("==============================")

test_inputs = np.array(
    [token_to_id["the"]],
    dtype=np.int64
)

logits, probabilities = inference(
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    test_inputs
)

print("Logits shape:", logits.shape)
print("Probabilities shape:", probabilities.shape)
print("Probabilities:", probabilities)

next_token_id = np.argmax(
    probabilities[-1]
)

print("Next token ID:", next_token_id)
print("Next token:", id_to_token[next_token_id])


print()
print("==============================")
print("PREFIX PREDICTION TEST")
print("==============================")

prefixes = [
    ["the"],
    ["the", "cat"],
    ["the", "cat", "sat"],
    ["the", "cat", "sat", "on"],
    ["the", "cat", "sat", "on", "the"]
]

for prefix in prefixes:

    prefix_ids = np.array(
        [token_to_id[token] for token in prefix],
        dtype=np.int64
    )

    logits, probabilities = inference(
        E,
        P,
        blocks,
        gamma_final,
        beta_final,
        W_lm,
        prefix_ids
    )

    next_token_id = np.argmax(probabilities[-1])
    next_token = id_to_token[next_token_id]

    print(
        f"{' '.join(prefix):25s}"
        f" -> {next_token:>3s}"
    )

print()
print("==============================")
print("GENERATION TEST")
print("==============================")

prompt_tokens = ["the"]

generated_tokens = generate(
    prompt_tokens,
    token_to_id,
    id_to_token,
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    max_new_tokens=5,
    max_context_length=max_context_length
)

print("Generated tokens:", generated_tokens)
print("Generated text:", " ".join(generated_tokens))
