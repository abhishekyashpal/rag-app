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

def transformer_block(
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

    H = Z @ block["W1"] + block["b1"]

    H = np.maximum(
        0,
        H
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

    return output


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

vocab_size = len(vocab)
embedding_dim = 4

E = np.random.randn(
        vocab_size,
        embedding_dim
    )
print('Embedding matrix', E.shape)

embeddings = E[token_ids]

print('Embeddings', embeddings)

max_context_length = 10

P = np.random.randn(
        max_context_length,
        embedding_dim
    )

print('Positional encoding matrix', P.shape)

sequence_length = len(token_ids)
print('Sequence length', sequence_length)
positions = np.arange(sequence_length)

print(positions)

position_embeddings = P[positions]

print('Position embeddings', position_embeddings.shape)

X = embeddings + position_embeddings

print('X', X.shape[1])

d_model = X.shape[1]

num_heads = 2

assert d_model % num_heads == 0

d_head = d_model // num_heads

print('d_model:', d_model)
print('d_head:', d_head)

d_ff = 16



num_layers = 3
blocks = []
for _ in range(num_layers):
    block = {
        "W_Q1": np.random.randn(d_model, d_head),
        "W_K1": np.random.randn(d_model, d_head),
        "W_V1": np.random.randn(d_model, d_head),
        "W_Q2": np.random.randn(d_model, d_head),
        "W_K2": np.random.randn(d_model, d_head),
        "W_V2": np.random.randn(d_model, d_head),
        "W_O": np.random.randn(d_model, d_model),
        "gamma1": np.ones(d_model),
        "beta1": np.zeros(d_model),
        "W1": np.random.randn(d_model, d_ff),
        "b1": np.zeros(d_ff),
        "W2": np.random.randn(d_ff, d_model),
        "b2": np.zeros(d_model),
        "gamma2": np.ones(d_model),
        "beta2": np.zeros(d_model)
    }
    blocks.append(block)


hidden = X

for block in blocks:
    hidden = transformer_block(hidden, block)

print("Hidden shape:", hidden.shape)

gamma_final = np.ones(d_model)
beta_final = np.zeros(d_model)

hidden_norm = layer_norm(hidden, gamma_final, beta_final)

print("Hidden norm shape:", hidden_norm.shape)

W_lm = np.random.randn(d_model, vocab_size)

print("W_lm shape:", W_lm.shape)

logits = hidden_norm @ W_lm

print("Logits shape:", logits)

inputs  = token_ids[:-1]
targets = token_ids[1:]
print("inputs:", inputs)
print("targets:", targets)
# print("input tokens:", [id_to_token[i] for i in inputs])
# print("target tokens:", [id_to_token[i] for i in targets])


probabilities = softmax(logits, axis=-1)
print("Probabilities shape:", probabilities)
print('probabilities:', np.sum(probabilities, axis=-1))


