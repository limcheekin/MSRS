# Summarize using local model e.g. Qwen2.5-7B
domain="story"
retrieval_setting="bm25"
model="qwen-7"
python3 summarize.py \
    --domain $domain \
    --source_file ../retrieval/$domain/$retrieval_setting/test.json \
    --output_dir ./$domain/$retrieval_setting/$model \
    --overwrite_output_dir \
    --model $model

# Summarize using API model e.g. Gemini 2.5 Pro
domain="meeting"
retrieval_setting="nv2-rerank"
model="gemini-2-5-pro"
python3 summarize_api.py \
    --domain $domain \
    --source_file ../retrieval/$domain/$retrieval_setting/test.json \
    --output_dir ./$domain/$retrieval_setting/$model \
    --overwrite_output_dir \
    --model $model

# Oracle setting
domain="meeting"
retrieval_setting="oracle"
model="llama-3-1-70"
python3 summarize.py \
    --domain $domain \
    --source_file ../retrieval/$domain/$retrieval_setting/test.json \
    --output_dir ./$domain/$retrieval_setting/$model \
    --overwrite_output_dir \
    --model $model \
    --use_oracle_prompt