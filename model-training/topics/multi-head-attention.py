import numpy as np

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

W_Q1 = np.random.randn(d_model, d_head)
W_K1 = np.random.randn(d_model, d_head)
W_V1 = np.random.randn(d_model, d_head)

W_Q2 = np.random.randn(d_model, d_head)
W_K2 = np.random.randn(d_model, d_head)
W_V2 = np.random.randn(d_model, d_head)

Q1 = X @ W_Q1
K1 = X @ W_K1
V1 = X @ W_V1

Q2 = X @ W_Q2
K2 = X @ W_K2
V2 = X @ W_V2

attention_scores1 = Q1@K1.T/np.sqrt(d_head)
attention_scores2 = Q2@K2.T/np.sqrt(d_head)

print(attention_scores1)
print(attention_scores2)

T = X.shape[0]
print('T', T)

mask = np.triu(
    np.ones((T, T)),
    k=1
)

print('Mask', mask)

masked_scores1 = np.where(
    mask == 1,
    -np.inf,
    attention_scores1
)

masked_scores2 = np.where(
    mask == 1,
    -np.inf,
    attention_scores2
)

print('Masked scores', masked_scores1)
print('Masked scores', masked_scores2)

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attention_weights1 = softmax(masked_scores1, axis=-1)
attention_weights2 = softmax(masked_scores2, axis=-1)

print('Attention weights', attention_weights1)
print('Attention weights', attention_weights2)

O1 = attention_weights1 @ V1
O2 = attention_weights2 @ V2

print("Q shape:", Q1.shape)
print("K shape:", K1.shape)
print("V shape:", V1.shape)
print("Scores shape:", attention_scores1.shape)
print("Masked scores shape:", masked_scores1.shape)
print("Attention shape:", attention_weights1.shape)
print("Output shape:", O1.shape)
print("Output shape:", O2.shape)

combined = np.concatenate(
    [O1, O2],
    axis=-1
)

W_O = np.random.randn(
    d_model,
    d_model
)

O = combined @ W_O

Y = X + O


print('X shape:', X.shape)
print('Combined output shape:', O.shape)
print('Y shape:', Y.shape)

def layer_norm(X, gamma, beta, eps=1e-5):

    mean = np.mean(
        X,
        axis=-1,
        keepdims=True
    )

    variance = np.var(
        X,
        axis=-1,
        keepdims=True
    )

    normalized = (
        X - mean
    ) / np.sqrt(
        variance + eps
    )

    output = gamma * normalized + beta

    return output

gamma = np.ones(d_model)
beta = np.zeros(d_model)

print('gamma shape:', gamma)
print('beta shape:', beta)

Z = layer_norm(
    Y,
    gamma,
    beta
)

print("Y shape:", Y.shape)
print("Z shape:", Z.shape)

d_ff = 16

W1 = np.random.randn(d_model, d_ff)
b1 = np.zeros(d_ff)

W2 = np.random.randn(d_ff, d_model)
b2 = np.zeros(d_model)


## First Linear Transformation
H = Z @ W1 + b1

##  RELU activation
H = np.maximum(0, H)        

## Second Linear Transformation
F = H @ W2 + b2

## Residual Connection
R = Z + F

## Second Layer Normalization
gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)

output = layer_norm(R, gamma2, beta2)

print("Output shape:", output.shape)
