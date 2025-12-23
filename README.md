## Setup

Experiments were performed on 2xT4 (Kaggle) environment with CUDA 12.3  
To run MMLU we used [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) project  
Due to extremely long duration of benchmark evaluation we truncated each task to 12 sapmles per task so there are 2736 prompts in total  
For calibration we used dataset open_platypus

## How to reproduce

Install (it will use python-venv)
```bash
sudo chmod +x setup_env.sh
sh setup_env.sh
```

Run experiments
```bash
python3 compress.py
```

To perform single experiment with metrics calculation
```bash
python3 evaluate.py --original-model <...> --compressed-model <...>
```

## Results

| Model | Size (Gb) | Performance (MMLU) | Compression Ratio | Performance Drop | Score |
|-------|-----------|-------------------|------------------|-----------------|-------|
| Qwen3-8B | 15,26 | 0,7617 | - | - | - |
| Qwen3-8B-GPTQ-W8A8 | 8,79 | 0,7412 | 1,74 | 0,027 | 1,6906 |
| Qwen3-8B-GPTQ-W4A16 | 5,65 | 0,7412 | 2,70 | 0,027 | 2,6301 |
| __Qwen3-8B-SmoothQuant-GPTQ-W4A16__ | 5,65 | 0,7462 | 2,70 | 0,020 | __2,6470__ |
| Qwen3-8B-BNB-INT8 | 8,79 | 0,7573 | 1,74 | 0,006 | 1,7261 |
| **Qwen3-8B-BNB-INT4** | 5,6 | 0,7398 | 2,73 | 0,029 | **2,6488** |

> **Note**:  
> - BNB - Bits-And-Bytes, means LLM.Int8() dynamic quantization.  
> - GPTQ and SmoothQuant are static quantization, so probably these models have lower latency, however only W8A8 quantization performs matmul in full int8 precision and perhaps the fastest one in compute-intensive case.