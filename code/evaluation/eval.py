import json
import multi_rouge
import os
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import bert_score
import asyncio
import aiolimiter
import math
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async def req(message):
        async with limiter:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=message,
                logprobs=True,
                top_logprobs=20,
                max_tokens=1
            )
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
    for metric_type in metric_list:
        for i in range(num_passes):
            # Get prompt
            prompt = open(f"GEval/prompts/{metric_type}_detailed.txt").read()
            # Get messages
            messages = []
            for index, prediction in enumerate(predictions):
                reference = references[index]
                cur_prompt = prompt.replace("{{Document}}", reference)
                cur_prompt = cur_prompt.replace("{{Summary}}", prediction)
                messages.append([{"role": "system", "content": cur_prompt}])

            # Get all G-Eval score responses
            response_list = await score_summaries(
                messages=messages,
                requests_per_minute=180
            )
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
        save_path = os.path.join(path, "evaluation", model_name, file_name)
        with open(save_path, "w") as f:
            json.dump(outputs, f)


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
    # Load predictions
    predictions = load_pred(path, model_name)
    if not predictions:
        return
    predictions = [
        predictions[index] for index, item in enumerate(predictions)
        if item != ""
    ]
    # Load references
    references = load_ref(dataset)
    references = [
        references[index] for index, item in enumerate(predictions) 
        if item != ""
    ]

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
            "bm25", "qwen-1-5", "nv2", "gemini-embedding", "oracle"
        ]
        models = [
            "llama2-7", "llama2-70", "llama3-1-8", "qwen-7", "llama3-3-70", 
            "llama3-1-70", "qwen-72", "gpt-4o-mini", "gpt-4o", "gemini-1-5-pro", 
            "gemini-2-flash", "deepseek-v3"
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
    perform_evaluations = False 
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
                    geval=True
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