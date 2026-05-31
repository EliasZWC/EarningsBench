# EarningsBench

This repository contains code and data samples for the paper:

> **EarningsBench: A Benchmark for Abnormal Optimistic Tone in Chinese Financial Reports**  
> *Submitted for double-blind review*

## Overview

EarningsBench is a benchmark dataset for fine-grained optimistic tone analysis in Chinese financial report Management Discussion and Analysis (MD&A) sections. The dataset contains 5,000 annotated sentences with optimism intensity labels (0-3 scale, inter-annotator Kappa = 0.847).

This repository supports:
- Sentence-level optimistic tone classification
- Abnormal Optimistic Tone (AOT) quantification (TOT - FOT)
- Fine-tuning of LLMs with chain-of-thought reasoning

## Repository Structure
```
earningsbench/
├── data/
│   ├── sample/           # 10% data sample (500 sentences)
│   ├── train.json        # Training split (70%)
│   ├── dev.json          # Development split (15%)
│   └── test.json         # Test split (15%)
├── scripts/
│   ├── evaluate.py       # Evaluation script
│   ├── baseline_lr.py    # Logistic Regression baseline
│   ├── baseline_bert.py  # BERT/FinBERT fine-tuning
│   └── baseline_llm.py   # LLM fine-tuning with CoT
├── requirements.txt
└── LICENSE
```

## Data Format

Each sample is a JSON object with the following fields:

{
  "sentence_id": "unique_identifier",
  "text": "The original sentence from MD&A section",
  "label": 0,
  "reason": "Automated reasoning annotation for CoT fine-tuning"
}

## Data Format

Each sample is a JSON object with the following fields:

{
  "sentence_id": "unique_identifier",
  "text": "The original sentence from MD&A section",
  "label": 0,
  "reason": "Automated reasoning annotation for CoT fine-tuning"
}

Label values:

0 = Neutral or Pessimistic

1 = Slightly Optimistic

2 = Relatively Optimistic

3 = Very Optimistic

## Quick Start

### Installation

Clone the repository and install dependencies:

```
git clone https://anonymous.4open.science/r/EarningsBench
cd EarningsBench
pip install -r requirements.txt
```

### Run Evaluation

Run the evaluation script on the test set:

python scripts/evaluate.py --model bert --data data/test.json

Available models: lr, xgboost, textcnn, bilstm, bert, finbert, llama, qwen

## License

This repository contains two types of content under different licenses.

Code (scripts, evaluation tools): MIT License

Data samples (annotated sentences in the data/ folder): CC BY 4.0 License

The original financial data used to construct this benchmark is sourced from CSMAR and CNRDS, and is subject to their respective license agreements. The annotations and data samples in this repository are original contributions of the authors.

See the LICENSE file in the root of this repository for the full license texts.

## Citation

If you use this benchmark in your research, please cite:

```
@article{earningsbench2026,
  title={EarningsBench: A Benchmark for Abnormal Optimistic Tone in Chinese Financial Reports},
  author={Anonymous for submission},
  year={2026}
}
```

## Contact

For questions during the review process, please contact the authors via the paper submission system.
