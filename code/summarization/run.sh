# Summarize using local model e.g. Qwen3-4B
export OPENAI_API_KEY=na
export GEMINI_PROJECT=na
export GEMINI_LOCATION=na
export LOCALAI_API_KEY=sk-1
export LOCALAI_BASE_URL=https://limcheekin--qwen3-4b-instruct-llama-server.modal.run/v1
export domain=story
export model=qwen3-4b-instruct
export retrieval_setting=qwen3-0-6
python3 summarize_api.py \
    --domain $domain \
    --source_file ../retrieval/$domain/$retrieval_setting/test.json \
    --output_dir ./$domain/$retrieval_setting/$model \
    --overwrite_output_dir \
    --model $model
