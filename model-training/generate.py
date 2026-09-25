import numpy as np

from .inference import inference


def generate(
    prompt_tokens,
    token_to_id,
    id_to_token,
    E,
    P,
    blocks,
    gamma_final,
    beta_final,
    W_lm,
    max_new_tokens,
    max_context_length
):
    # Convert prompt tokens to token IDs
    token_ids = [
        token_to_id[token]
        for token in prompt_tokens
    ]

    for _ in range(max_new_tokens):

        # Keep only the most recent context
        context_ids = token_ids[-max_context_length:]

        inputs = np.array(
            context_ids,
            dtype=np.int64
        )

        # Run inference
        logits, probabilities = inference(
            E,
            P,
            blocks,
            gamma_final,
            beta_final,
            W_lm,
            inputs
        )

        # Get probabilities for the last token
        next_token_probabilities = probabilities[-1]

        # Greedy decoding:
        # choose the most probable next token
        next_token_id = np.argmax(
            next_token_probabilities
        )

        # Append generated token
        token_ids.append(
            int(next_token_id)
        )

    # Convert IDs back to tokens
    generated_tokens = [
        id_to_token[token_id]
        for token_id in token_ids
    ]

    return generated_tokens