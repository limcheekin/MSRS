# Sparse retriever e.g. BM25
domain=story
python3 retrieve.py \
    --domain $domain \
    --queries_path ../../data/$domain/queries_test.json \
    --documents_path ../../data/$domain/documents \
    --embeddings_path embeddings/$domain/bm25 \
    --method bm25 \
    --model bm25

# Dense retriever e.g. NV-Embed-v2, gemini-embedding
domain=story
python3 retrieve.py \
    --domain $domain \
    --queries_path ../../data/$domain/queries_test.json \
    --documents_path ../../data/$domain/documents \
    --embeddings_path embeddings/$domain/nv2 \
    --method llm_embedding \
    --model nv2

domain=story
python3 retrieve.py \
    --domain $domain \
    --queries_path ../../data/$domain/queries_test.json \
    --documents_path ../../data/$domain/documents \
    --embeddings_path embeddings/$domain/gemini-embedding \
    --method llm_embedding \
    --model gemini-embedding

# Other splits e.g. dev
domain=story
python3 retrieve.py \
    --domain $domain \
    --queries_path ../../data/$domain/queries_dev.json \
    --documents_path ../../data/$domain/documents \
    --embeddings_path embeddings/$domain/bm25_dev \
    --method bm25 \
    --model bm25_dev \
    --split dev