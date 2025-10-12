./bin/llama-server -m ./models/Qwen3-Embedding-0.6B-f16.gguf \
               --embedding \
               --host 0.0.0.0 \
               --port 8886 \
               --pooling last \
#               -ub 8192
