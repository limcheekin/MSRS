import os
import json
from dotenv import load_dotenv
load_dotenv()
import aiolimiter
from openai import AsyncOpenAI
from google import genai
from google.genai import types
import asyncio
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
gclient = genai.Client(
    vertexai=True, 
    project=os.environ["GEMINI_PROJECT"],
    location=os.environ["GEMINI_LOCATION"],
    http_options={"api_version": "v1"}
)
limiter = aiolimiter.AsyncLimiter(200)

async def generate(messages, id, meeting_ids, query, model_name, save_dir):
    async def req(message):
        async with limiter:  
            if model_name == "gpt-4o-mini":
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=message,
                    top_p=0.9,
                    temperature=0.7,
                    max_tokens=10
                )
                return response.choices[0].message.content
            elif model_name == "gemini-2-flash":
                response = gclient.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=message,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.9,
                        max_output_tokens=10,
                    ),
                )
                return response.text
        
    async_responses = [req(message) for message in messages]
    responses = await tqdm_asyncio.gather(*async_responses)
    output = []
    for i in range(len(meeting_ids)):
        output.append((meeting_ids[i], responses[i]))
    with open(f"{save_dir}/{id}.json", "w") as f:
        json.dump({
            "query": query,
            "relevance_scores": output
        }, f)

async def main(rerank_dict, domain, model_name, save_dir):
    id = -1
    for query in rerank_dict:
        id += 1
        if os.path.exists(f"{save_dir}/{id}.json"):
            continue
        prompts = []
        for meeting_id in rerank_dict[query]:
            with open(f"../../data/{domain}/documents/{meeting_id}.txt") as f:
                doc = f.read()
                if domain == "meeting":
                    prompt = f"""You are given a question about a discussion that happened over multiple meetings. You are also given the transcript of a meeting. Your task is to assign a score from 1 to 20 based on how relevant this specific meeting transcript is to answering the question. Respond with only a numerical score.

    QUESTION: {query}

    MEETING: {doc}
    """
                elif domain == "story":
                    prompt = f"""You are given a question about an unknown story and a chapter from a specific story. Your task is to assign a score from 1 to 20 based on how relevant this chapter is to answering the question. Respond only with a numerical score.

    QUESTION: {query}

    CHAPTER: {doc}
    """
                if model_name == "gpt-4o-mini":
                    prompts.append([{"role": "user", "content": prompt}])
                elif model_name == "gemini-2-flash":
                    prompts.append(prompt)
        await generate(
            prompts, id, rerank_dict[query], query, model_name, save_dir
        )

if __name__ == "__main__":
    domain = "story"
    retrieval_setting = "qwen-1-5"
    rerank_path = f"rerank/{domain}/{retrieval_setting}"
    model_name = "gemini-2-flash"
    with open(f"rankings.json") as f:
        rerank_dict = json.load(f)
    asyncio.run(main(rerank_dict, domain, model_name, rerank_path))

    query_to_ranking = {}
    print(len(os.listdir(rerank_path)))
    for i in range(len(os.listdir(rerank_path))):
        try:
            with open(f"{rerank_path}/{i}.json") as f:
                entry = json.load(f)
            relevance_scores = list(
                map(lambda x: (int(x[1]), x[0]), entry["relevance_scores"])
            )
            for i in range(len(relevance_scores)):
                relevance_scores[i] = (
                    relevance_scores[i][0],
                    -(i + 1), 
                    relevance_scores[i][1]
                )
            relevance_scores.sort(reverse=True)
            query_to_ranking[entry["query"]] = relevance_scores
        except:
            print("Error", i)
            break
    with open(f"{rerank_path}/rankings.json", "w") as f:
        json.dump(query_to_ranking, f)