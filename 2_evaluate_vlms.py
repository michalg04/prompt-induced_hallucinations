from general_utils import *
import argparse
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig,Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
)
import pandas as pd
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Run VLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next', 'qwen' ,'janus', 'llava-one'], required=True, help="Choose the model version.")
    parser.add_argument('--dataset_size', type=str, choices=['full', 'mini'], required=True)
    parser.add_argument('--task', type=str, choices=['count', 'color'], required=True)
    args = parser.parse_args()
    
    model_version=args.model_version  

    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    if model_version == 'llava-one':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True)
        
        model = model.eval()

        
    elif model_version == 'janus':
        model_path = "deepseek-ai/Janus-Pro-7B"
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = model.to(torch.bfloat16).cuda().eval()
    
    elif model_version == 'qwen':
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
    
    if args.task == "color":
        dataset = load_dataset("mgolov/Visual-Counterfact")
    
        df = dataset["color"].to_pandas()

        valid_colors = {'red','orange','yellow','green','blue','purple'}
        df = df[
            df["correct_answer"].apply(
                lambda x: ast.literal_eval(x)[0] in valid_colors
            )
        ]

    else:
        #Generate this file using 1_create_counting_dataset.ipynb
        df = pd.read_csv("counting_with_prompts.csv")
    
    if args.dataset_size == 'mini':
        df=df.head(5)
    
    if args.task == "color":
        df = eval_color_pih(df, processor, model, model_version, args.task)
        df = eval_color_base(df, processor, model, model_version, args.task)
    else:
        df["correct_answer"] = df["number"]
        df = eval_counts(df, processor, model, model_version, args.task)

    df.to_csv(f'{model_version}_{args.task}_results_{args.dataset_size}.csv', index=False)

if __name__ == "__main__":
    main()