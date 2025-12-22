from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bit', action='store', choices=['4', '8'])
    parser.add_argument('outdir', action='store', type=str)
    args = parser.parse_args()

    if args.bit == '8':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                                 llm_int8_enable_fp32_cpu_offload=True)
    elif args.bit == '4':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model8bit = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config
    )
    
    model8bit.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)

if __name__ == '__main__':
    main()
