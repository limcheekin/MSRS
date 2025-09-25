# Dense retriever e.g. NV-Embed-v2, gemini-embedding
export OPENAI_API_KEY=na
export GEMINI_PROJECT=na
export GEMINI_LOCATION=na
export LOCALAI_API_KEY=sk-1
export LOCALAI_BASE_URL=http://192.168.1.111:8880/v1
export domain=story
export model=qwen3-0-6
python3 retrieve.py \
    --domain $domain \
    --queries_path ../../data/$domain/queries_test.json \
    --documents_path ../../data/$domain/documents \
    --embeddings_path embeddings/$domain/$model \
    --method llm_embedding \
    --model $model
