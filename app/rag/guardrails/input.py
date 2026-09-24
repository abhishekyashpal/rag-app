MAX_QUERY_LENGTH = 2000


def validate_query(query: str) -> str:
    if query is None:
        raise ValueError("Query cannot be None.")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters."
        )

    return query

def normalize_query(query: str) -> str:
    query = query.strip()

    # Collapse repeated whitespace
    query = " ".join(query.split())

    return query

def validate_and_normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    query = normalize_query(query)

    if not query:
        raise ValueError("Query cannot be empty.")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError("Query is too long.")

    return query