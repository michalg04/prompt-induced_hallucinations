from general_utils import *
from knockout_utils import *
import argparse
import pandas as pd
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig,Qwen2VLForConditionalGeneration,LlavaOnevisionForConditionalGeneration
)
import torch
import argparse
from PIL import Image
import numpy as np
import pickle

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


def get_image_token_indices(model_version, input_ids, tokenizer):
    if model_version == "llava-one":
        img_token_id = tokenizer.convert_tokens_to_ids("<image>")
        img_indices = np.where(input_ids == img_token_id)[0]
        text_indices = np.where(input_ids != img_token_id)[0]
        return img_indices, text_indices

    elif model_version == "qwen":
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vision_start_pos = np.where(input_ids == vision_start)[0]
        vision_end_pos = np.where(input_ids == vision_end)[0]
        if len(vision_start_pos) == 0 or len(vision_end_pos) == 0:
            raise ValueError("Could not find <|vision_start|> or <|vision_end|> tokens")
        img_indices = np.arange(vision_start_pos[0], vision_end_pos[0] + 1)
        text_indices = np.setdiff1d(np.arange(len(input_ids)), img_indices)
        return img_indices, text_indices

    elif model_version == "janus":
        img_token_id = tokenizer.convert_tokens_to_ids("<image_placeholder>")
        img_indices = np.where(input_ids == img_token_id)[0]
        text_indices = np.where(input_ids != img_token_id)[0]
        return img_indices, text_indices

    else:
        raise ValueError(f"Unknown model_version: {model_version}")
        
def run_head_knockout_sweep_att_mass(model, model_version, processor, image_path, prompt,
                            max_new_tokens=10, do_sample=False, temperature=0.0,
                            limit_layers=None, limit_heads=None, heads=None):

    inputs = build_inputs(image_path, model_version, processor, prompt, model)

    seqs, attns_base = generate_text(
        model, processor, inputs, model_version,
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, return_attn=True
    )
    
    if model_version == "janus": 
        tok = processor.tokenizer
        baseline_text = tok.decode(seqs[0].tolist(), skip_special_tokens=True)
        img_indices, text_indices = get_image_token_indices(
            model_version, 
            inputs["input_ids"], 
            processor.tokenizer
        )
        
    else:
        baseline_text = processor.batch_decode(seqs, skip_special_tokens=True)[0]
        img_indices, text_indices = get_image_token_indices(
            model_version, 
            inputs.input_ids[0].cpu().numpy(), 
            processor.tokenizer
        )
        

    baseline_attn_mass_layer = compute_attention_mass(attns_base, img_indices, text_indices)

    if model_version == "qwen":
        num_layers = model.config.num_hidden_layers
        num_heads  = model.config.num_attention_heads
    elif model_version == "janus":
        num_layers = len(model.language_model.model.layers)
        num_heads  = model.config.language_config.num_attention_heads
    else:
        num_layers = len(model.language_model.model.layers)
        num_heads  = model.config.text_config.num_attention_heads

    layer_iter = range(num_layers) if limit_layers is None else limit_layers
    head_iter  = range(num_heads)  if limit_heads  is None else limit_heads

    zeroer = HeadMeanAblation(model, model_version)
    changes = []

    all_abl_layer_masses = []
    
    for L in layer_iter:
        for H in head_iter:
            
            with zeroer.knockout(heads): 
                seqs, attns_abl = generate_text(
                    model, processor, inputs, model_version,
                    max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=temperature, return_attn=True
                )
                if model_version == "janus":
                    ablated = processor.tokenizer.decode(
                        seqs[0].tolist(), skip_special_tokens=True
                        )
                else:
                    ablated = processor.batch_decode(seqs, skip_special_tokens=True)[0]

            # Per-layer and per-head
            abl_layer_mass = compute_attention_mass(attns_abl, img_indices, text_indices)
            all_abl_layer_masses.append(abl_layer_mass)
            
            changed = (ablated != baseline_text)
            entry = {"layer": L, "head": H, "ablated_text": ablated, "changed": changed}
            if changed:
                entry["diff"] = diff_strings(baseline_text, ablated)
            changes.append(entry)

            print(f"[L{L:02d} H{H:02d}] {'CHANGED' if changed else 'same'}")


    ablated_attn_mass_layer = {}
    for layer in baseline_attn_mass_layer.keys():
        img_vals = [m[layer]["img"] for m in all_abl_layer_masses]
        txt_vals = [m[layer]["text"] for m in all_abl_layer_masses]
        ablated_attn_mass_layer[layer] = {
            "img": float(np.mean(img_vals)),
            "text": float(np.mean(txt_vals))
        }

    
    return (
        baseline_text,
        ablated,
        changes,
        baseline_attn_mass_layer,
        ablated_attn_mass_layer
    )


def compute_attention_mass(attentions, img_indices, text_indices):
    layer_dict = {}

    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx].squeeze().detach().cpu().to(torch.float32).numpy()  # [H, T, T]
        attn_from_last_token = attn[:, -1, :]  # [H, T]

        attn_to_img = attn_from_last_token[:, img_indices]
        img_mass = np.sum(attn_to_img) / attn_to_img.shape[0] if attn_to_img.ndim == 2 else float(attn_to_img)

        attn_to_text = attn_from_last_token[:, text_indices]
        text_mass = np.sum(attn_to_text) / attn_to_text.shape[0] if attn_to_text.ndim == 2 else float(attn_to_text)

        layer_dict[layer_idx] = {"img": img_mass, "text": text_mass}

    return layer_dict
    

    
def main():
    parser = argparse.ArgumentParser(description="Compute attention mass shift to image tokens on LLAVA, Qwen, and janus models.")
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
    

    df= pd.read_csv(f"correct_{args.model_version}_{args.task}_{args.dataset_size}.csv")
    
    
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
    per_layer_results = []
    
    
    for row in df.reset_index().itertuples(index=False):
        ridx = row.index
        image_path   = row.image_path
        #to reduce memory use
        if args.model_version == "llava-one":
            img = Image.open(image_path)
            # Get size
            width, height = img.size
            if width > 600 or height > 600:
                print(f"Image size for IDX {ridx}: {width} x {height}")
                continue 
                
        prompt  = row.prompt

        if args.evaluation == True:
            (
                baseline_text,
                ablated_text,
                changes,
                base_mass_layer,
                abl_mass_layer
            ) = run_head_knockout_sweep_att_mass(
                model, args.model_version, processor, image_path, prompt,
                max_new_tokens=10, do_sample=False, temperature=0.0,
                limit_layers=[1], limit_heads=[1], heads=heads_to_ablate
            )
        else:
            (
                baseline_text,
                ablated_text,
                changes,
                base_mass_layer,
                abl_mass_layer
            ) = run_head_knockout_sweep_att_mass(
                model, args.model_version, processor, image_path, prompt,
                max_new_tokens=10, do_sample=False, temperature=0.0,
                limit_layers=LIMIT_LAYERS, limit_heads=LIMIT_HEADS, heads=heads_to_ablate
            )
    
        
        per_layer_results.append({
            "sample_index": ridx,
            "image_path": image_path,
            "prompt": prompt,
            "baseline_text": baseline_text,
            "ablated_text": ablated_text,
            "baseline_attn_mass": base_mass_layer,
            "ablated_attn_mass": abl_mass_layer,
        })
    
    
    with open(f"attn_mass_per_layer_{args.model_version}_{args.heads_to_ablate}.pkl", "wb") as f:
        pickle.dump(per_layer_results, f)
    


if __name__ == "__main__":
    main()
