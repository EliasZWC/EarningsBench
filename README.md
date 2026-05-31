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
├── data/                         # 10% data sample (500 sentences total)
│   ├── train_samples.json        # Training split (70% of the 10%, with reason)
│   ├── dev_samples.json          # Development split (15% of the 10%, with reason)
│   └── test_samples.json         # Test split (15% of the 10%, without reason)
├── scripts/
│   ├── train_bert.py             # BERT-base (Chinese)
│   ├── train_cnn.py              # TextCNN
│   ├── train_finbert.py          # FinBERT (Chinese)
│   ├── train_finscot.py          # LLM + FinSCoT
│   ├── train_lora.py             # LLM + LoRA
│   └── train_lstm.py             # BiLSTM
├── requirements.txt
└── LICENSE
```

## Data Format

Each sample is a JSON object with the following fields:

```json
{
  "instruction": "请判断以下财报文本句子，判断其表达的情感标签（0-中性或悲观, 1-稍微乐观, 2-比较乐观, 3-非常乐观）。",
  "input": "The original sentence from MD&A section",
  "output": "1",
  "reason": "Automated reasoning annotation for FinSCoT fine-tuning"
}
```

> Note: The "reason" field is only present in training and development sets. Test set samples contain only instruction, input, and output fields.

Label values:

- 0 = Neutral or Pessimistic
- 1 = Slightly Optimistic
- 2 = Relatively Optimistic
- 3 = Very Optimistic

## Quick Start

### Installation

Clone the repository and install dependencies:
```
git clone https://anonymous.4open.science/r/EarningsBench
cd EarningsBench
pip install -r requirements.txt
```

### Run Baseline Models

Run BERT-base (Chinese) on the sample data:
```
python scripts/train_bert.py --data_dir data/
```

Run FinBERT on the sample data:
```
python scripts/train_finbert.py --data_dir data/
```

Run TextCNN on the sample data:
```
python scripts/train_cnn.py --data_dir data/
```

Run BiLSTM on the sample data:
```
python scripts/train_lstm.py --data_dir data/
```

Run LLM + LoRA (baseline without CoT):
```
python scripts/train_lora.py --data_dir data/ --model_dir /path/to/llm
```

Run FinSCoT (LLM + FinSCoT, our method):
```
python scripts/train_finscot.py --data_dir data/ --model_dir /path/to/llm
```

> Note: For LLM-based models (train_lora.py and train_finscot.py), you need to download the base LLM (e.g., Qwen3-8B or LLaMA3.1-8B) and specify its path via --model_dir.

## License

This repository contains two types of content under different licenses.

Code (scripts in scripts/ folder): MIT License

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
