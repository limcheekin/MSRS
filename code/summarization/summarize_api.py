import json
import os
import argparse
import time
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT_ORACLE_STORY = "You are a helpful assistant. After reading a lengthy story, provide a 200 to 300 word answer to a question posed about the story. Directly respond to the question and do not chat with the user."
SYSTEM_PROMPT_STORY = "You are a helpful assistant. After reading chapters from various stories, provide a 200 to 300 word answer to a question posed about a specific story. Directly respond to the question and do not chat with the user."
SYSTEM_PROMPT_MEETING = "You are a helpful assistant. After reading a set of lengthy meeting transcripts, provide a 200 to 300 word answer to a question posed about the meetings. Directly respond to the question and do not chat with the user."
SYSTEM_PROMPT_CONTAMINATION_CHECK_STORY = "You are a helpful assistant who has seen the following story from the SQuALITY dataset before. Provide a 200 to 300 word answer to a question posed about the story. Rely upon your internal knowledge of the story and give it your best guess."
SYSTEM_PROMPT_CONTAMINATION_CHECK_MEETING = "You are a helpful assistant who has seen the meeting transcripts from the QMSum dataset before. Provide a 200 to 300 word answer to a question posed about these meetings. Rely upon your internal knowledge of the meetings and give it your best guess."

def load_data_from_path(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_prompt_no_chat(instruction, system_prompt=SYSTEM_PROMPT_STORY):
    prompt_template =  system_prompt + "\n" + instruction 
    return prompt_template

def cut_off_text(text, substr):
    index = text.find(substr)
    if index != -1:
        return text[:index]
    else:
        return text

def generate_and_save_summaries(
    client,
    model,
    data,
    output_folder,
    domain,
    use_oracle_prompt=False,
    use_contamination_check_prompt=False
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
        if count >= stop: break
        if os.path.exists(os.path.join(output_folder, f"{count}.txt")):
            continue
        print(count)
        query = item["Query"]
        # Truncation
        # truncate = False
        # word_limit = 20000
        # if domain == "meeting":
        #     truncate = True
        article = item["Article"]
        # if truncate:
        #     docs = item["Article"].split("<doc-sep>")
        #     doc_words = [doc.split(" ") for doc in docs]
        #     print(list(map(len, doc_words)))
        #     article = "<doc-sep>".join([" ".join(x[:word_limit // len(docs)]) for x in doc_words])
        print(len(article), len(article.split(" ")))
        # Prompt
        if use_contamination_check_prompt:
            if domain == "meeting":
                formatted_instruction = f"Answer the following question posed about the meetings. QUESTION: {query}\nANSWER: "
            else:
                # item["Title"] contains both the story title and author
                # For example, "Raiders of the Second Moon by GENE ELLERMAN"
                formatted_instruction = f"Answer the following question posed about the story {item['Title']}. QUESTION: {query}\nANSWER: "
        else:
            if domain == "meeting":
                formatted_instruction = f"Answer the following question posed about the meetings.\nQUESTION: {query}\nMEETINGS: {article}\nANSWER: "
            else:
                formatted_instruction = f"Answer the following question posed about a story.\nQUESTION: {query}\nCHAPTERS: {article}\nANSWER: "
        # Choose appropriate system prompt
        system_prompt = SYSTEM_PROMPT_MEETING if domain == "meeting" else SYSTEM_PROMPT_STORY
        if use_oracle_prompt and domain == "story":
            system_prompt = SYSTEM_PROMPT_ORACLE_STORY
        elif use_contamination_check_prompt:
            if domain == "meeting":
                system_prompt = SYSTEM_PROMPT_CONTAMINATION_CHECK_MEETING
            elif domain == "story":
                system_prompt = SYSTEM_PROMPT_CONTAMINATION_CHECK_STORY
        # Generate response through model API
        prompt = get_prompt_no_chat(formatted_instruction, system_prompt)
        start_time = time.time()
        summary = None
        if model in ["gpt-4o-mini", "gpt-4o"]:
            # OpenAI API
            response = client.chat.completions.create(
                model = model,
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.7, 
                top_p=0.9, 
                max_tokens=600,
            )
            summary = response.choices[0].message.content
        elif model in ["gpt-5", "gpt-5-mini", "gpt-5-nano", "o3"]:
            # OpenAI API
            response = client.chat.completions.create(
                model = model,
                messages=[{"role": "user", "content": prompt}], 
                temperature=1
            )
            summary = response.choices[0].message.content
        # REF: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF    
        elif model in ["qwen3-4b"]:
            # LocalAI API
            response = client.chat.completions.create(
                model = model,
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.7, 
                top_p=0.8,
                max_tokens=600,
            )
            summary = response.choices[0].message.content            
        elif model in ["deepseek-v3", "gpt-oss-20b", "gpt-oss-120b", "deepseek-r1"]:
            # Fireworks AI API
            if model == "deepseek-v3":
                response = client.chat.completions.create(
                    model = f"accounts/fireworks/models/{model}",
                    messages=[{"role": "user", "content": prompt}], 
                    temperature=0.7, 
                    top_p=0.9, 
                    max_tokens=600,
                )
            else:
                api_model = model
                if model == "deepseek-r1":
                    api_model = "deepseek-r1-0528"
                response = client.chat.completions.create(
                    model = f"accounts/fireworks/models/{api_model}",
                    messages=[{"role": "user", "content": prompt}], 
                )
            summary = response.choices[0].message.content
        elif model in [
            "gemini-1-5-pro", "gemini-2-flash", "gemini-2-5-flash", "gemini-2-5-pro"
        ]:
            api_names = {
                "gemini-1-5-pro": "gemini-1.5-pro",
                "gemini-2-flash": "gemini-2.0-flash",
                "gemini-2-5-flash": "gemini-2.5-flash",
                "gemini-2-5-pro": "gemini-2.5-pro"
            }
            # Gemini API
            if model in ["gemini-1-5-pro", "gemini-2-flash"]:
                response = client.models.generate_content(
                    model=api_names[model],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.9,
                        max_output_tokens=600,
                    ),
                )
            elif model in ["gemini-2-5-flash", "gemini-2-5-pro"]:
                response = client.models.generate_content(
                    model=api_names[model],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.9
                    ),
                )
            summary = response.text
        with open(os.path.join(output_folder, f"{count}.txt"), "w") as f:
            f.write(summary)
        elapsed_time = round(time.time() - start_time, 2)
        print("Took", elapsed_time, "seconds")

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
    parser = argparse.ArgumentParser(description="Summarization")
    parser.add_argument("--domain", type=str, default="story", choices=["meeting", "story"],)
    parser.add_argument("--source_file", type=str, required=True, help="Path to the input data file (test.json)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated summaries")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--model", type=str, required=True, choices=[
        "gpt-4o-mini", "gpt-4o", "gemini-1-5-pro", "gemini-2-flash", 
        "deepseek-v3", "deepseek-r1", "gemini-2-5-flash", "gemini-2-5-pro", 
        "gpt-oss-20b", "gpt-oss-120b", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o3",
        "qwen3-4b"
    ])
    parser.add_argument("--use_oracle_prompt", action="store_true")
    parser.add_argument("--use_contamination_check_prompt", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.overwrite_output_dir:
        print(f"Overwriting contents in {args.output_dir}")

    data = load_data_from_path(args.source_file)

    client = None
    if args.model in [
        "gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o3"
    ]:
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
    elif args.model in ["qwen3-4b"]:
        client = client = OpenAI(
            base_url="https://limcheekin--qwen3-4b-instruct-llama-server.modal.run/v1",
            api_key=os.environ["LOCALAI_API_KEY"],
            timeout=900 # 15 minutes
        )        
    elif args.model in ["deepseek-v3", "deepseek-r1", "gpt-oss-20b", "gpt-oss-120b"]:
        client = client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FW_API_KEY"]
        )
    elif args.model in ["gemini-1-5-pro", "gemini-2-flash", "gemini-2-5-flash", "gemini-2-5-pro"]:
        client = genai.Client(
            vertexai=True,
            project=os.environ["GEMINI_PROJECT"],
            location=os.environ["GEMINI_LOCATION"],
            http_options={"api_version": "v1"}
        )
    generate_and_save_summaries(
        client,
        args.model,
        data,
        args.output_dir,
        args.domain,
        args.use_oracle_prompt,
        args.use_contamination_check_prompt
    )

if __name__ == "__main__":
    main()
