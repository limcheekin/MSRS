import json
import os
import argparse
import torch
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

model_to_path = {
    "llama2-7": "unsloth/llama-2-7b-chat-bnb-4bit",
    "llama2-70": "iarbel/Llama-2-70b-chat-hf-bnb-4bit",
    "llama3-1-8": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama3-1-70": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
    "llama3-3-70": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    "qwen-7": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen-72": "unsloth/Qwen2.5-72B-Instruct-bnb-4bit"
}
supported_models = model_to_path.keys()

B_INST, E_INST = "<s>[INST] ", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\\n", "\\n<</SYS>>\\n\\n"
SYSTEM_PROMPT_ORACLE_STORY = "You are a helpful assistant. After reading a lengthy story, provide a 200 to 300 word answer to a question posed about the story. Directly respond to the question and do not chat with the user."
SYSTEM_PROMPT_STORY = "You are a helpful assistant. After reading chapters from various stories, provide a 200 to 300 word answer to a question posed about a specific story. Directly respond to the question and do not chat with the user."
SYSTEM_PROMPT_MEETING = "You are a helpful assistant. After reading a set of lengthy meeting transcripts, provide a 200 to 300 word answer to a question posed about the meetings. Directly respond to the question and do not chat with the user."

def load_data_from_path(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_prompt(instruction, system_prompt=SYSTEM_PROMPT_STORY):
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def get_prompt_no_chat(instruction, system_prompt=SYSTEM_PROMPT_STORY):
    prompt_template =  system_prompt + "\n" + instruction 
    return prompt_template

def generate_and_save_summaries(
    llm,
    data,
    domain,
    output_folder,
    model_name,
    use_oracle_prompt=False
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start = 0
    stop = len(data)
    count = -1
    print(f"Generate summaries from {start} to {stop}:")
    for item in data:
        count += 1
        if count < start: continue
        if count >= stop: continue
        if os.path.exists(os.path.join(output_folder, f"{count}.txt")):
            with open(os.path.join(output_folder, f"{count}.txt")) as f:
                text = f.read()
                if len(text) > 0:
                    continue
        print(count)
        query = item["Query"]
        print(count, query)
        # Truncation
        truncate = False
        word_limit = 20000
        if domain == "meeting":
            truncate = True
        if model_name in ["llama2-7", "llama2-70"]:
            truncate = True
            word_limit = 2000
        article = item["Article"]
        if truncate:
            docs = item["Article"].split("<doc-sep>")
            doc_words = [doc.split(" ") for doc in docs]
            print(list(map(len, doc_words)))
            article = "<doc-sep>".join([" ".join(x[:word_limit // len(docs)]) for x in doc_words])
        print(len(article), len(article.split(" ")))
        # Prompt
        if domain == "meeting":
            formatted_instruction = f"""Answer the following question posed about the meetings.
QUESTION: {query}
MEETINGS: {article}
ANSWER:
"""
        elif domain == "story":
            formatted_instruction = f"""Answer the following question posed about a story.
QUESTION: {query}
CHAPTERS: {article}
ANSWER:
"""
        system_prompt = SYSTEM_PROMPT_MEETING if domain == "meeting" else SYSTEM_PROMPT_STORY
        if use_oracle_prompt and domain == "story":
            system_prompt = SYSTEM_PROMPT_ORACLE_STORY
        prompt = (
            get_prompt(formatted_instruction, system_prompt)
            if "chat" in model_to_path[model_name]
            else get_prompt_no_chat(formatted_instruction, system_prompt)
        )
        params = SamplingParams(
            temperature=0.7, 
            top_p=0.9, 
            max_tokens=600,
            repetition_penalty=1.1
        )
        output = llm.generate(prompt, params)
        text = output[0].outputs[0].text
        with open(os.path.join(output_folder, f"{count}.txt"), "w") as f:
            f.write(text)

    contents = os.listdir(output_folder)
    for i in range(len(data)):
        if not f"{i}.txt" in contents:
            print("Not all summaries have been generated so will not save to JSON yet...")
            return

    results = []
    for i in range(len(data)):
        with open(os.path.join(output_folder, f"{i}.txt")) as f:
            text = f.read()
            if len(text) == 0:
                print("Empty -", i)
            results.append(text)
    save_path = os.path.join(output_folder, "generated_summaries.json")
    print(f"Saved summaries to JSON at {save_path}")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Run LlaMA for text summarization")
    parser.add_argument("--source_file", type=str, required=True, help="Path to the input data file (test.json)")
    parser.add_argument("--domain", type=str, choices=["story", "meeting"])
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated summaries")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--use_oracle_prompt", action="store_true")
    parser.add_argument("--model", type=str, required=True, choices=supported_models)
    args = parser.parse_args()
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.overwrite_output_dir:
        print(f"Overwriting contents in {args.output_dir}")
    data = load_data_from_path(args.source_file)
    # Initialize model and generate summaries
    print("Loading", model_to_path[args.model])
    max_model_len = 32000
    if args.model in ["llama2-7", "llama2-70"]:
        max_model_len = 4000
    llm = LLM(
        model=model_to_path[args.model],
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enforce_eager=True,
        device="cuda"
    )
    generate_and_save_summaries(
        llm,
        data,
        args.domain,
        args.output_dir,
        args.model,
        args.use_oracle_prompt
    )

if __name__ == "__main__":
    main()
