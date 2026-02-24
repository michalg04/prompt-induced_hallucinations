
from general_utils import *
from knockout_utils import *

import argparse
import ast
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
)

from janus.models import MultiModalityCausalLM, VLChatProcessor



def run_attention_sample_analysis(model, model_version, processor, image_path, prompt, row, task='count', heads_to_save=None):
    results = {}
    tokenizer=processor.tokenizer            
    inputs = build_inputs(image_path, model_version, processor, prompt, model, task=task, device="cuda")

    if model_version == "janus":
        # INPUT IDS VERSION.-- Janus outputs attentions for just the last sequences! This is fine now that we have changed stuff. 
        outputs = model.language_model.generate(
            inputs_embeds=inputs['inputs_embeds'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=10,
            num_beams=1,
            do_sample=False,
            use_cache=False,
            temperature=1.0,
            output_attentions=False,
            return_dict_in_generate=True,
         )

        output_seq = outputs.sequences         # full sequence of token ids
        
        # ---- FIX: build FULL inputs_embeds ----
        # concat inputs and reponse

        response = clean_answer(tokenizer.decode(output_seq.squeeze()), model_version)    
        prompt = prompt + ' ' + response
        print("NEW PROMPT: ", prompt)


        inputs = build_inputs(image_path, model_version, processor, prompt, model, task=task, device="cuda")

        print("IDS SHAPE: ", inputs['inputs_embeds'].shape)

        # ---- FINAL FULL FORWARD PASS ----
        outputs = model.language_model(
            inputs_embeds=inputs['inputs_embeds'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True,
            use_cache=False,
        )

    # -------------------------------------------------
    # QWEN / Qwen2-VL
    # -------------------------------------------------
    elif model_version == "qwen":
        outputs = model.generate(
                    **inputs,
                    max_new_tokens=10, # CHANGING FOR NOW 10,
                    do_sample=False,
                    use_cache=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

        
        generated_ids = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
            ][0]
        
        output_seq = outputs.sequences

    # -------------------------------------------------
    # LLaVA-family, InternVL, etc.
    # -------------------------------------------------
    else:
        outputs = model.generate(**inputs, max_new_tokens=15, num_beams=1, do_sample=False, temperature=1.0, return_dict_in_generate=True)
        output_seq = outputs.sequences         # full sequence of token ids
        
        # ---- FIX: build FULL inputs_embeds ----
        # concat inputs and reponse

        response = clean_answer(tokenizer.decode(output_seq.squeeze()), model_version)    
        prompt = prompt + ' ' + response
        print("NEW PROMPT: ", prompt)


        inputs = build_inputs(image_path, model_version, processor, prompt, model, task=task, device="cuda")
        outputs = model(**inputs, output_attentions=True)
        attn = outputs.attentions
        print("LEN ATTNL ", len(attn))
        print("SHAPE: ", attn[0].shape)

            
    if model_version=='janus':
        inputs = inputs['inputs_ids']
    else:
        inputs = inputs.input_ids.squeeze()        

    if model_version in ['llava-one', 'janus']:
        # Janus: outputs.attentions is already a list of length num_layers
        # each item has shape (1, heads, seq, seq)
        all_layer_attns = outputs.attentions
        
        attn_all = np.stack([
            layer.squeeze(0)                        # remove batch dim
                 .detach()
                 .to(torch.float32)
                 .cpu()
                 .numpy()
            for layer in all_layer_attns
        ], axis=0).astype(np.float16)  # shape = (layers, heads, seq, seq)

        print("ATTN ALL SHAPE: ", attn_all.shape)

    else:
        # normal HF logic
        all_layer_attns = outputs.attentions
        last_step_attentions = outputs.attentions[-1]  # This is a tuple of tensors (one per layer)
        
        # Now stack them into a numpy array
        attn_all = np.stack([
            layer.squeeze(0).detach().cpu().to(torch.float32).numpy()
            for layer in last_step_attentions
        ], axis=0)
        
    # attn_all -- this is where we need to do our indexing. 
    attn_results = {}
    for tup in heads_to_save:
        layer = tup[0]
        head = tup[1]
        attn_results[f'{layer}_{head}'] = attn_all[layer, head, :, :]   
    
    results["answer"] = output_seq.detach().cpu().to(torch.float32).numpy()
    results["text"] = clean_answer(tokenizer.decode(output_seq.squeeze()), model_version)
    results["input_ids"] = inputs.detach().cpu().to(torch.float32).numpy()
    results["attn"] = attn_results

    # --- Cleanup ---
    for var in ['outputs', 'inputs', 'image', 'prepare_inputs']:
        if var in locals():
            del locals()[var]
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run attention output on LLAVA, Qwen, and Janus models for color or counting tasks.")
    parser.add_argument('--model_version', type=str, choices=[ 'qwen','llava-one', 'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=[ 'count', 'color'], required=True, help="Choose the task.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")
    parser.add_argument("--heads_to_ablate", type=str, default=None, help="Top 1/2/3.")
    parser.add_argument("--num_samples", type=str, default="20")
    args = parser.parse_args()
    
    model_version=args.model_version    

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    
    if model_version == 'llava-one':
        # SHIFT TO LLAVA-ONE
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        
        # load your model normally
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        )
        
        # recursively set all attention layers to eager
        def set_eager_attention(module):
            for child in module.children():
                if hasattr(child, "set_attn_implementation"):
                    child.set_attn_implementation("eager")
                set_eager_attention(child)
        
        print("LOAD IN FILE PATH FOR LLAVA=ONE")
        
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

    heads_to_ablate = head_ablate_dict[args.model_version][args.heads_to_ablate]
    
    print(heads_to_ablate)

    attention_results = {}
    
    if args.num_samples == 'all':
        sample_df = df
    elif args.num_samples.startswith('[') and args.num_samples.endswith(']'):
        # Convert string "[15,30]" to a Python list [15, 30]
        start, end = ast.literal_eval(args.num_samples)
        sample_df = df.iloc[start:end]
    else:
        sample_df = df.head(int(args.num_samples))
        

    # print("WARNING: RUNNING BOTH HEADS TO ABLATE AND BASELINE")
    print("Running baseline")
            
    for row in sample_df.reset_index().itertuples(index=False):
        i = row.index
        print(i)
        image_path   = row.path
        img = Image.open(image_path)

        img = Image.open(image_path)
    
        # Get size
        width, height = img.size
        if width > 600 or height > 600:
            print(f"Image size for IDX {i}: {width} x {height}")
            continue 
        
        prompt  = row.first_wrong_prompt
        row = row._asdict()

        print(f" BASE IDX: {i}, Prompt: {prompt}")

        result = run_attention_sample_analysis(model, model_version, processor, image_path, prompt, row, 
                                                task=args.task, heads_to_save=heads_to_ablate)
        attention_results[i]=result
    
    if args.num_samples == 'all':   
        print("SAVING")
        file_path = f"attention_results_{model_version}_{args.task}_{heads_to_ablate}_baseline.pkl"
        print(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(attention_results, f) 
    else:
        print("SAVING")
        file_path = f"attention_results_{model_version}_{args.task}_{heads_to_ablate}_baseline_{args.num_samples}.pkl"
        print(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(attention_results, f) 
            
if __name__ == '__main__':
    main()




