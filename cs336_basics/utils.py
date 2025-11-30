import regex as re

def process_chunk_freqs(chunk_data, special_tokens, PAT):
    start, end, file_path = chunk_data

    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    
    special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
    segments = re.split(f"({special_tokens_pattern})", chunk)

    token_freqs = {}

    special_token_bytes = {token.encode('utf-8') for token in special_tokens}

    for segment in segments:
        if not segment or segment in special_tokens:
            continue
        
        for match in re.finditer(PAT, segment):
            token = match.group(0)
            token_bytes = token.encode('utf-8')
            if token_bytes in special_token_bytes:
                continue
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            token_freqs[token_tuple] = token_freqs.get(token_tuple, 0) + 1

    return token_freqs