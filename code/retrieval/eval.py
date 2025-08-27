import json
import math
import os
from collections import defaultdict

class InformationRetrieval:
    def __init__(
        self, 
        query_file, 
        meeting_folder, 
    ):
        with open(query_file, "r") as file:
            self.queries_dict = json.load(file)
        self.results = {"MIN": [], "MEAN": [], "MAX": []}
        self.performance_metrics = {"MIN": [], "MEAN": [], "MAX": []}
        self.meeting_ids = []
        self.meetings = []

        for filename in sorted(os.listdir(meeting_folder)):
            meeting_id = filename.split(".")[0]  # Assuming filename is like "TS3009d.txt"
            self.meeting_ids.append(meeting_id)

            with open(os.path.join(meeting_folder, filename), "r") as file:
                self.meetings.append(file.read())

        # Read meetings and create a mapping from meeting IDs to texts
        self.meeting_texts = {}
        for filename in os.listdir(meeting_folder):
            meeting_id = filename.split(".")[0]
            with open(os.path.join(meeting_folder, filename), "r") as file:
                self.meeting_texts[meeting_id] = file.read()

    # Function to calculate performance metrics
    def evaluate_performance(self, retrieved_meetings, ground_truth_meetings):
        retrieved_set = set(retrieved_meetings)
        ground_truth_set = set(ground_truth_meetings)

        tp = len(retrieved_set.intersection(ground_truth_set))
        fp = len(retrieved_set) - tp
        fn = len(ground_truth_set) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        # NDCG calculation
        dcg = 0
        ideal_dcg = 0
        for i, doc in enumerate(retrieved_meetings):
            rel = 1 if doc in ground_truth_set else 0
            dcg += rel / math.log2(i + 2)  # i starts from 0, log starts from 2
        for i in range(min(len(ground_truth_meetings), len(retrieved_meetings))):
            ideal_dcg += 1 / math.log2(i + 2)
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        # MAP calculation (Average Precision for this query)
        num_relevant = 0
        sum_precision = 0
        for i, doc in enumerate(retrieved_meetings):
            if doc in ground_truth_set:
                num_relevant += 1
                sum_precision += num_relevant / (i + 1)
        ap = sum_precision / len(ground_truth_set) if ground_truth_set else 0
        return {"Precision": precision, "Recall": recall, "F1": f1, "NDCG": ndcg, "AP": ap}

    def run_evaluation(
        self, 
        rankings,
        n_values=[1, 3, 6], 
        labels=["MIN", "MEAN", "MAX"]
    ):
        # Initialize dictionary to store sum and count for each metric and label
        for n, label in zip(n_values, labels):
            performance_metrics = defaultdict(
                lambda: defaultdict(lambda: {"sum": 0.0, "count": 0})
            )
            # Initialize list to hold results for this method and top_k
            for i, query_data in enumerate(self.queries_dict.values()):
                ground_truth_meetings = query_data["gold_documents"]
                retrieved_meetings = rankings[i]
                # Evaluate performance
                metrics = self.evaluate_performance(retrieved_meetings, ground_truth_meetings)
                # Accumulate for average metrics
                for metric_name, metric_value in metrics.items():
                    performance_metrics[label][metric_name]["sum"] += metric_value
                    performance_metrics[label][metric_name]["count"] += 1

        for label, metrics in performance_metrics.items():
            print(f"Performance Metrics for {label}({n}):")
            for metric_name, values in metrics.items():
                avg_value = values["sum"] / values["count"]
                print(f"{metric_name}: {avg_value:.4f}")
            print()


if __name__ == "__main__":
    domain = "meeting" # Choose either "meeting" or "story"
    retrieval_settings = [
        "bm25", "bm25-rerank", "promptriever", "qwen-1-5", "qwen-1-5-rerank",
        "qwen-7", "gritlm", "nv1", "nv2", "nv2-rerank", "text-3-small",
        "text-3-large", "gemini-embedding", "gemini-embedding-rerank"
    ]
    queries_path = f"../../data/{domain}/queries_test.json"
    documents_path = f"../../data/{domain}/documents"
    n = 8 if domain == "story" else 3
    for split in retrieval_settings:
        with open(f"{domain}/{split}/test.json") as f:
            retrieved = json.load(f)
        rankings = [entry["Ranking"] for entry in retrieved]
        ir = InformationRetrieval(queries_path, documents_path)
        print(domain, split)
        ir.run_evaluation(
            rankings,
            [n],
            ["MEAN"]
        )