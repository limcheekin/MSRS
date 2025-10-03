import json
import multi_rouge
import os
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import bert_score
import asyncio
import aiolimiter
import math
import random
import re
from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

async def make_openai_request_with_retry(message, max_retries=5, base_delay=1):
    """Make OpenAI API request with exponential backoff retry logic.

    Args:
        message: The message to send to OpenAI API
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 1)

    Returns:
        OpenAI API response object

    Raises:
        RateLimitError: If rate limit is hit after all retries
        Exception: If other API errors occur after all retries
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=message,
                logprobs=True,
                top_logprobs=20,
                max_tokens=1
            )
            return response

        except RateLimitError as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise

            # Extract wait time from error message if available
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            if "Please try again in" in str(e):
                try:
                    # Extract milliseconds from error message
                    match = re.search(r'Please try again in (\d+)ms', str(e))
                    if match:
                        wait_time = max(wait_time, int(match.group(1)) / 1000.0)
                except Exception:
                    pass

            print(f"Rate limit hit, waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
            await asyncio.sleep(wait_time)

        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise

            wait_time = base_delay * (2 ** attempt)
            print(f"API error: {type(e).__name__}: {e}, waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
            await asyncio.sleep(wait_time)

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in retry logic")

def compute_geval_score(top_logprobs):
    """Computes G-Eval score given top log probabilities of next token."""
    target_tokens = {"1", "2", "3", "4", "5"}
    logprob_dict = {
        e.token: e.logprob for e in top_logprobs if e.token in target_tokens
    }
    prob_dict = {
        token: math.exp(logprob) for token, logprob in logprob_dict.items()
    }
    for token in target_tokens:
        prob_dict.setdefault(token, 0.0)
    # Normalize probabilities
    total_prob = sum(prob_dict.values())
    prob_dict = {token: prob / total_prob for token, prob in prob_dict.items()}
    # Compute weighted average
    geval_score = sum(float(token) * prob for token, prob in prob_dict.items())
    return geval_score


async def score_summaries(messages, requests_per_minute):
    """Computes G-Eval scores for all the summaries in the given list of
    OpenAI API messages, using request throttling.
    """
    # Use both request-based and token-based limiting
    # Estimate ~500 tokens per request (conservative estimate)
    tokens_per_minute = 25000  # Conservative limit below 30k TPM
    estimated_tokens_per_request = 500
    max_requests_by_tokens = tokens_per_minute // estimated_tokens_per_request

    # Use the more restrictive limit
    effective_rpm = min(requests_per_minute, max_requests_by_tokens)
    print(f"Using effective rate limit: {effective_rpm} requests per minute")

    limiter = aiolimiter.AsyncLimiter(effective_rpm)
    async def req(message):
        async with limiter:
            response = await make_openai_request_with_retry(message)
            top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            return compute_geval_score(top_logprobs)

    async_responses = [req(message) for message in messages]
    responses = await tqdm_asyncio.gather(*async_responses)
    return responses


def load_pred(path, model_name):
    """Loads prediction summaries in the current directory for the specified 
    model.
    """
    predictions = []
    pred_file = f"{model_name}_summary.json"
    file_path = os.path.join(path, "summary", pred_file)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            predictions = json.load(f)
    return predictions


def load_ref(dataset: str):
    """Loads reference summaries for the given dataset."""
    dataset = dataset.lower()
    if dataset == "story":
        ref_file = "../../data/story/oracle/test.json"
    elif dataset == "meeting":
        ref_file = "../../data/meeting/oracle/test.json"
    references = []
    if os.path.exists(ref_file):
        with open(ref_file, "r") as f:
            references = json.load(f)
        if dataset == "story":
            references = [
                [
                    data_item["Summary_1"],
                    data_item["Summary_2"],
                    data_item["Summary_3"],
                    data_item["Summary_4"],
                ]
                for data_item in references
            ]
        elif dataset == "meeting":
            references = [data_item["Summary"] for data_item in references]
    return references


def compute_average(x):
    """Compute the average of a list."""
    if len(x) == 0:
        return 0
    return sum(x) / len(x)


def evaluate_rouge(path, predictions, references, model_name, dataset):
    """Conducts ROUGE evaluation for the specified model with the given
    prediction and reference summaries.

    Uses MultiROUGE since ODSum-Story and SQuALITY have four reference summaries
    per query. Also works for ODSum-Meet with one reference summary.
    """
    print("Evaluate rouge score")
    rouge_object = multi_rouge.Rouge()
    squality_rouge_score = []
    dataset = dataset.lower()
    if dataset == "story":
        squality_rouge_score = rouge_object._compute(
            predictions=predictions, references=references, use_stemmer=True
        )
    elif dataset == "meeting":
        squality_rouge_score = rouge_object._compute(
            predictions=predictions,
            references=[[item] for item in references],
            use_stemmer=True,
        )
    file_name = f"{model_name}_squality_rouge.json"
    file_path = os.path.join(path, "evaluation", model_name, file_name)
    rouge_scores = json.loads(json.dumps(squality_rouge_score))
    modes = ["low", "mid", "high"]
    for key in rouge_scores:
        entry = {}
        for i in range(3):
            e = rouge_scores[key][i]
            entry[modes[i]] = {
                "precision": e[0],
                "recall": e[1],
                "f1-measure": e[2]
            }
        rouge_scores[key] = entry
    with open(file_path, "w") as f:
        json.dump(rouge_scores, f)


def evaluate_bert(path, predictions, references, model_name):
    """Conducts BERTScore evaluation for the specified model with the given 
    prediction and reference summaries."""
    print("Evaluate bert score")
    batch_size = 261
    bert_scores = {
        "p": [],
        "r": [],
        "f1": [],
        "average_p": 0,
        "average_r": 0,
        "average_f1": 0,
    }
    num_batches = (len(predictions) + batch_size - 1) // batch_size  

    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min(start + batch_size, len(predictions))

        pred_batch = predictions[start:end]
        ref_batch = references[start:end]

        p, r, f1 = bert_score.score(pred_batch, ref_batch, lang="en")
        # Add in bert_scores
        for index in range(len(p)):
            bert_scores["r"].append(float(p[index]))
            bert_scores["p"].append(float(r[index]))
            bert_scores["f1"].append(float(f1[index]))

    # Calculate average bert
    average_p = compute_average(bert_scores["p"])
    average_r = compute_average(bert_scores["r"])
    average_f1 = compute_average(bert_scores["f1"])
    bert_scores["average_p"] = average_p
    bert_scores["average_r"] = average_r
    bert_scores["average_f1"] = average_f1
    # Save
    file_name = f"{model_name}_bert_score.json"
    file_path = os.path.join(path, "evaluation", model_name, file_name)
    with open(file_path, "w") as f:
        temp = json.dumps(bert_scores)
        f.write(temp)


async def evaluate_geval(
    path, predictions, references, model_name, dataset, geval_summary_index=1
):
    """Conducts G-Eval evaluation for the specified model with the given
    predicition and reference summaries."""
    passes = []
    outputs = {}
    num_passes = 1
    metric_list = ["rel"]
    dataset = dataset.lower()

    # Check for existing partial results
    save_path = os.path.join(path, "evaluation", model_name)
    progress_file = os.path.join(save_path, f"{model_name}_geval_progress.json")

    for metric_type in metric_list:
        for i in range(num_passes):
            # Check if this pass was already completed
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                        if f"{metric_type}_pass_{i}" in progress:
                            print(f"Resuming from saved progress for {metric_type} pass {i}")
                            response_list = progress[f"{metric_type}_pass_{i}"]
                            average_score = compute_average(response_list)
                            passes.append(average_score)

                            # Store pass information
                            geval = {}
                            if dataset == "story":
                                geval["Summary_" + str(geval_summary_index)] = response_list
                            elif dataset == "meeting":
                                geval["Summary"] = response_list
                            geval["Average"] = average_score
                            outputs[f"Pass #{i + 1}"] = geval
                            continue
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load progress file: {e}. Starting fresh.")

            # Get prompt
            prompt = open(f"GEval/prompts/{metric_type}_detailed.txt").read()
            # Get messages
            messages = []
            for index, prediction in enumerate(predictions):
                reference = references[index]
                cur_prompt = prompt.replace("{{Document}}", reference)
                cur_prompt = cur_prompt.replace("{{Summary}}", prediction)
                messages.append([{"role": "system", "content": cur_prompt}])

            print(f"Processing {len(messages)} messages for {metric_type} pass {i}")
            # Get all G-Eval score responses
            response_list = await score_summaries(
                messages=messages,
                requests_per_minute=30  # Reduced from 180 to avoid rate limits
            )

            # Save progress
            progress = {}
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not read existing progress file: {e}")
                    progress = {}

            progress[f"{metric_type}_pass_{i}"] = response_list

            try:
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
                print(f"Progress saved for {metric_type} pass {i}")
            except IOError as e:
                print(f"Warning: Could not save progress file: {e}")
            # Calculate average 
            average_score = compute_average(response_list)
            passes.append(average_score)
            # Store pass information
            geval = {}
            if dataset == "story":
                geval["Summary_" + str(geval_summary_index)] = response_list
            elif dataset == "meeting":
                geval["Summary"] = response_list
            geval["Average"] = average_score
            outputs[f"Pass #{i + 1}"] = geval
        # Store overall information
        average = compute_average(passes)
        outputs["Scores"] = [sorted(passes)]
        outputs["Average Score"] = [average]
        outputs["Average Percentage Score"] = average * 20
        file_name = f"{model_name}_{metric_type}_gpteval.json"
        final_save_path = os.path.join(path, "evaluation", model_name, file_name)
        with open(final_save_path, "w") as f:
            json.dump(outputs, f)

        # Clean up progress file on successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"Evaluation completed successfully. Cleaned up progress file.")


async def evaluate_model(
    path,
    model_name,
    dataset: str,
    bert=False,
    rouge=False,
    geval=False,
    geval_summary_index=1,
):
    """Perform evaluations for the specified model at the given path."""
    print('load_pred(path, model_name)', path, model_name)
    # Load predictions
    predictions = load_pred(path, model_name)
    if not predictions:
        return
    predictions = [
        predictions[index] for index, item in enumerate(predictions)
        if item != ""
    ]
    print('predictions', predictions)

    # Load references
    references = load_ref(dataset)
    references = [
        references[index] for index, item in enumerate(predictions) 
        if item != ""
    ]
    print('references', references)

    # Create save directory if needed
    save_path = os.path.join(path, "evaluation", model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if rouge:
        evaluate_rouge(
            path, 
            predictions, 
            references, 
            model_name, 
            dataset
        )
    if bert:
        evaluate_bert(path, predictions, references, model_name) 
    if geval:
        if dataset.lower() == "story":
            references = [
                item[geval_summary_index - 1] for item in references
            ]
        await evaluate_geval(
            path,
            predictions,
            references,
            model_name,
            dataset,
            geval_summary_index,
        )


async def main():
    # Choose which domain, retrieval settings, and models to evaluate
    domain = "story" # Either "meeting" or "story"
    if domain == "story":
        retrieval_settings = [
            # "bm25", "qwen-1-5", "nv2", "gemini-embedding", "oracle"
            "qwen3-0-6"
        ]
        models = [
            # "llama2-7", "llama2-70", "llama3-1-8", "qwen-7", "llama3-3-70", 
            # "llama3-1-70", "qwen-72", "gpt-4o-mini", "gpt-4o", "gemini-1-5-pro", 
            # "gemini-2-flash", "deepseek-v3"
            "qwen3-4b"
        ]
    elif domain == "meeting":
        retrieval_settings = [
            "bm25", "bm25-rerank", "nv2", "nv2-rerank", "oracle"
        ]
        models = [
            "llama2-7", "llama2-70", "llama3-1-8","llama3-1-70", "llama3-3-70", 
            "qwen-7", "qwen-72", "gemini-1-5-pro", "gemini-2-flash", 
            "deepseek-v3", "gpt-4o-mini", "gpt-4o"
        ]
    # Flags
    perform_evaluations = True 
    # Compute averages across models (row) and retrieval settings (column) for
    # each metric
    row_averages = {}
    col_averages = {}
    metrics = ["rouge_1", "rouge_2", "bertscore", "geval"]
    for metric in metrics:
        row_averages[metric] = {}
        col_averages[metric] = {}
    for split in retrieval_settings:
        for metric in metrics:
            if not split in col_averages[metric]:
                col_averages[metric][split] = []
        for model in models:
            for metric in metrics:
                if not model in row_averages[metric]:
                    row_averages[metric][model] = []
            geval_path = (
                f"{domain}/{split}/evaluation/{model}/{model}_rel_gpteval.json"
            )
            if perform_evaluations and not os.path.exists(geval_path):
                await evaluate_model(
                    path=f"{domain}/{split}",
                    model_name=model,
                    dataset=domain, 
                    rouge=True,
                    bert=True,
                    geval=True,
                    geval_summary_index=2,
                )
            
            print(split, model)
            rouge_path = (
                f"{domain}/{split}/evaluation/{model}/{model}_squality_rouge.json"
            )
            with open(rouge_path) as f:
                rouge_scores = json.load(f)
                # ROUGE-1
                rouge_1_score = float(rouge_scores["rouge1"]["high"]["f1-measure"])
                rouge_1_score = round(rouge_1_score * 100, 2)
                print("ROUGE-1 F1 Score:", rouge_1_score)
                col_averages["rouge_1"][split].append(rouge_1_score)
                if split != "oracle":
                    row_averages["rouge_1"][model].append(rouge_1_score)
                # ROUGE-2
                rouge_2_score = float(rouge_scores["rouge2"]["high"]["f1-measure"])
                rouge_2_score = round(rouge_2_score * 100, 2)
                print("ROUGE-2 F1 Score:", rouge_2_score)
                col_averages["rouge_2"][split].append(rouge_2_score)
                if split != "oracle":
                    row_averages["rouge_2"][model].append(rouge_2_score)

            bertscore_path = (
                f"{domain}/{split}/evaluation/{model}/{model}_bert_score.json"
            )
            with open(bertscore_path) as f:
                bertscore = round(json.load(f)["average_f1"] * 100, 2)
                print("BERTScore F1 Score:", bertscore)
                col_averages["bertscore"][split].append(bertscore)
                if split != "oracle":
                    row_averages["bertscore"][model].append(bertscore)

            geval_path = (
                f"{domain}/{split}/evaluation/{model}/{model}_rel_gpteval.json"
            )
            with open(geval_path) as f:
                geval_score = round(json.load(f)["Average Percentage Score"], 2)
                print("G-Eval:", geval_score)
                col_averages["geval"][split].append(geval_score)
                if split != "oracle":
                    row_averages["geval"][model].append(geval_score)

            print("\n")
    
    # Compute column and row averages
    C = {}
    R = {}
    for metric in metrics:
        items = col_averages[metric].items()
        C[metric] = {
            split: round(compute_average(values), 2) for split, values in items
        }
        items = row_averages[metric].items()
        R[metric] = {
            model: round(compute_average(values), 2) for model, values in items
        }
    # Compute column average of row averages
    A = {}
    for metric in metrics:
        values = list(R[metric].values())
        A[metric] = round(compute_average(values), 2)

    print("Column averages:")
    for split in retrieval_settings:
        print(f"  - {split}: ROUGE-1 {C['rouge_1'][split]}, ROUGE-2 {C['rouge_2'][split]}, BERTScore {C['bertscore'][split]}, GEval {C['geval'][split]}")
    print("Row averages:")
    for model in models:
        print(f"  - {model}: ROUGE-1 {R['rouge_1'][model]}, ROUGE-2 {R['rouge_2'][model]}, BERTScore {R['bertscore'][model]}, GEval {R['geval'][model]}")
    print("Column average of row averages:")
    print(f"  - ROUGE-1 {A['rouge_1']}, ROUGE-2 {A['rouge_2']}, BERTScore {A['bertscore']}, GEval {A['geval']}")

asyncio.run(main())
