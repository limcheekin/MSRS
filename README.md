# MSRS: Evaluating Multi-Source Retrieval-Augmented Generation

#### [**📄 Paper**](https://arxiv.org/abs/2508.20867) | [**🤗 Data**](https://huggingface.co/datasets/yale-nlp/MSRS)

## Overview

This paper introduces a scalable framework for constructing evaluation benchmarks that challenge RAG systems to integrate information across distinct sources and generate long-form responses. Using our framework, we build two new benchmarks on Multi-Source Retrieval and Synthesis: MSRS-Story and MSRS-Meet.

<p align="center">
  <img src="./images/main-figure.png" alt="image-20230930180329172" width="70%" />
</p>

### Dataset Statistics

<p align="center">
  <img src="images/dataset-statistics.png" alt="image-20230930180308855" width="50%;" />
</p>

### Repository Structure

The datasets for MSRS-Story and MSRS-Meet are provided in the `data` directory. 

The retrieval code and the settings created by each retrieval model, which serve as inputs for summarization, are located in the `code/retrieval` directory.

The summarization code is included in `code/summarization`.

The evaluation code, along with the generated summaries and their corresponding evaluation results (e.g., ROUGE-2, G-Eval), are located in the `code/evaluation` directory.

## Quickstart

### 1. Setup

Install the required packages using Python version >=3.9. 

```
pip install -r requirements.txt
```

### 2. Run

Examples for running the retrieval, summarization, and evaluation scripts are provided in `usage.sh` files alongside the scripts.


## Experimental Results

Retrieval Peformance for MSRS-Story

<p align="center">
  <img src="./images/retrieval-story.png" alt="image-20230930180329172" width="70%" />
</p>

Retrieval Peformance for MSRS-Meet

<p align="center">
  <img src="./images/retrieval-meeting.png" alt="image-20230930180329172" width="70%" />
</p>

Summarization Performance for MSRS-Story

<p align="center">
  <img src="./images/summarization-story.png" alt="image-20230930180329172" width="70%" />
</p>


Summarization Performance for MSRS-Meet

<p align="center">
  <img src="./images/summarization-meeting.png" alt="image-20230930180329172" width="70%" />
</p>

Oracle Summarization Performance for Reasoning Models

<p align="center">
  <img src="./images/summarization-reasoning.png" alt="image-20230930180329172" width="70%" />
</p>

## Citation

If you find our work helpful, please consider citing it:

```bibtex
@inproceedings{
    phanse2025msrs,
    title={{MSRS}: Evaluating Multi-Source Retrieval-Augmented Generation},
    author={Rohan Phanse and Yijie Zhou and Kejian Shi and Wencai Zhang and Yixin Liu and Yilun Zhao and Arman Cohan},
    booktitle={Second Conference on Language Modeling},
    year={2025},
    url={https://openreview.net/forum?id=KtGsJm8bOC}
}
```