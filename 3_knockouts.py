from general_utils import *
from knockout_utils import *
import argparse
import pandas as pd
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, Qwen2VLForConditionalGeneration,LlavaOnevisionForConditionalGeneration
)
import torch
import argparse

# JANUS IMPORTS
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor



def main():
    parser = argparse.ArgumentParser(description="Run knockouts on LLAVA, Qwen, and Janus models for color or counting tasks.")
    parser.add_argument('--model_version', type=str, choices=[ 'qwen','llava-one', 'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=[ 'count', 'color'], required=True, help="Choose the task.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")
    parser.add_argument('--evaluation', type=str, choices=[ True, False], required=True, help="Choose whether you're evaluating knockout or running the full sweep.")
    #If doing full sweep, evaluation = False, use the following:
    parser.add_argument( "--layer_lower", type=int, default=None, help="Lower bound (inclusive) for layer index")
    parser.add_argument("--layer_upper", type=int, default=None, help="Upper bound (exclusive) for layer index")
    parser.add_argument( "--head_lower", type=int, default=None, help="Lower bound (inclusive) for layer index")
    parser.add_argument("--head_upper", type=int, default=None, help="Upper bound (exclusive) for layer index")
    #If doing only doing evaluation, evaluation = True, use the following:
    parser.add_argument("--heads_to_ablate", type=str, default=None, help="Top 1/2/3.")
    
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

        
    model_version=args.model_version  

    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    if model_version == 'llava-one':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True)
        
        
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
    
    df= pd.read_csv(f'{model_version}_{args.task}_results_{args.dataset_size}.csv')
    df = make_vertical_df(df)
    
    if args.dataset_size == "mini":
        df = df.head(5)    

    if args.layer_lower is None:
        LIMIT_LAYERS = None
    else:
        LIMIT_LAYERS = range(args.layer_lower, args.layer_upper)
        
    if args.head_lower is None:
        LIMIT_HEADS  = None
    else:
        LIMIT_HEADS  = range(args.head_lower, args.head_upper)

    if args.evaluation == True:
        heads_to_ablate = head_ablate_dict[args.model_version][args.heads_to_ablate]
    else:
        heads_to_ablate = None
    

    print(heads_to_ablate)
        
    all_rows = []  
    for row in df.reset_index().itertuples(index=False):
        ridx = row.index  
        image_path   = row.path
        ground_truth = row.correct_answer
        prompt  = row.prompt
        baseline_text = row.response
        
        if args.evaluation == True:
            changes = run_head_knockout_sweep(
                args.task, model, args.model_version, processor, image_path, prompt, baseline_text, 
                max_new_tokens=50, do_sample=False, temperature=0.0,
                limit_layers=[1], limit_heads=[1], heads=heads_to_ablate
            )
        else:
            changes = run_head_knockout_sweep(
                args.task, model, args.model_version, processor, image_path, prompt,baseline_text, 
                max_new_tokens=50, do_sample=False, temperature=0.0,
                limit_layers=LIMIT_LAYERS, limit_heads=LIMIT_HEADS, heads=heads_to_ablate
            )
    
        df_heads = summarize_head_changes(args.task, args.model_version,
            baseline_text, changes, ground_truth=ground_truth, include_diff=False
        )
    
        chg_df = pd.DataFrame(changes)[["layer", "head", "ablated_text"]]
        df_heads = df_heads.merge(chg_df, on=["layer", "head"], how="left")
        df_heads["baseline_text"]   = baseline_text
        df_heads["image_path"]      = image_path
        df_heads["ground_truth"]    = ground_truth
        df_heads["prompt"] = prompt
        df_heads["sample_index"]    = ridx  # keep track of which df row this came from
    
        all_rows.append(df_heads)
    
        print(f"[ok] row {ridx}: {len(df_heads)} head rows")


    # 6) concatenate all samples into one table
    all_df_heads = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if args.evaluation == True:
        all_df_heads.to_csv(f"best_PIH_ablation_results_{args.heads_to_ablate}_{args.model_version}_{args.task}_{args.dataset_size}.csv") 

    else:
        all_df_heads.to_csv(f"all_PIH_ablation_results_L{args.layer_lower}H{args.head_lower}_to_L{args.layer_upper}H{args.head_upper}_{args.model_version}_{args.task}_{args.dataset_size}.csv") 

    
if __name__ == "__main__":
    main()
