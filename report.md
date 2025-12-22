## Setup

Experiments were performed on 2xT4 (Kaggle) environment with CUDA 12.3
To run MMLU we used [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) project
Due to extremely long duration of benchmark evaluation we truncated each task to 12 sapmles per task so there are 2736 prompts in total

## How to reproduce

Install
```bash
pip install -r pyproject.toml
```
In case there is an error related to dependencies mismatch please run `pip install --upgrade compressed-tensors` one more time.

```bash
python3 compress.py
```

To perform single experiment with metrics calculation
```bash
python3 evaluate.py --path-to-original-model <...> --path-to-compressed-model <...>
```
