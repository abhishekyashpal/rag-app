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

d_k = d_model

W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_k)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

attention_scores = Q@K.T/np.sqrt(d_k)

print(attention_scores)

T = X.shape[0]
print('T', T)

mask = np.triu(
    np.ones((T, T)),
    k=1
)

print('Mask', mask)

masked_scores = np.where(
    mask == 1,
    -np.inf,
    attention_scores
)

print('Masked scores', masked_scores)

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attention_weights = softmax(masked_scores, axis=-1)

print('Attention weights', attention_weights)

O = attention_weights @ V

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("Scores shape:", attention_scores.shape)
print("Masked scores shape:", masked_scores.shape)
print("Attention shape:", attention_weights.shape)
print("Output shape:", O.shape)
