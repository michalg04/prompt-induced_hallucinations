from general_utils import *
from knockout_utils import *
import argparse
import pickle
from collections import Counter
from contextlib import contextmanager

# Third-party libraries
import torch
import pandas as pd
from num2words import num2words
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration, BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration)

from janus.models import MultiModalityCausalLM, VLChatProcessor





# NOT DE-COUPLED
# def process_results(results, cb_num, gt_num, processor, task='color', is_negated=False):

#     if task == 'color':
#         gt_num = ' ' + gt_num
#         cb_num = ' ' + cb_num
    
#     print("CB NUM: ", cb_num)
#     print("GT NUM: ", gt_num)
    
#     # THIS IS NOT GETTING NON-NUMERICAL RESPONSES. 
    
#     #STACK scores then get PROBABILITIES
#     scores = torch.stack(results[0]["scores"])                # [seq_len, 1, vocab]
#     scores = torch.stack([torch.softmax(s, dim=-1) for s in scores], dim=0)  
#     scores = scores.squeeze(1)                                # -> [seq_len, vocab]

#     vocab_size = scores.size(-1)

#     def tokenize_to_ids(num_str):
#         """Tokenize and return all valid token ids (skip specials)."""
#         ids = processor.tokenizer(num_str, add_special_tokens=False).input_ids
#         return [i for i in ids if i < vocab_size]

#     def expand_forms(num):
#         """Expand number into digit and word forms (always <=19)."""
#         n = int(num)
#         return [' ' + str(n), ' ' + num2words(n)]

    
#     if task == 'count':
#         # Expand cb and gt into digit+word forms
#         cb_forms = expand_forms(cb_num)
#         gt_forms = expand_forms(gt_num)
        
#     elif task == 'color':
#         cb_forms = [cb_num]
#         gt_forms = [gt_num]

#     # print("CB FORMS: ", cb_forms)
#     # print("GT FORMS: ", gt_forms)
    
#     # Convert forms to token IDs
#     cb_ids_all = []
#     for f in cb_forms:
#         cb_ids_all.extend(tokenize_to_ids(f))

#     gt_ids_all = []
#     for f in gt_forms:
#         gt_ids_all.extend(tokenize_to_ids(f))
 
#     # Count elements
#     cb_counter = Counter(cb_ids_all)
#     gt_counter = Counter(gt_ids_all)

#     # print("CB Count ", cb_counter)
#     # print("GT Count ", gt_counter)
    
#     # Subtract counts (only positive counts remain)
#     cb_ids = list((cb_counter - gt_counter).elements())
#     gt_ids = list((gt_counter - cb_counter).elements())

#     # print("CB ids: {}. Decoded{} ".format(cb_ids, processor.tokenizer.convert_ids_to_tokens(cb_ids)))
#     # print("GT ids: {}. Decoded{} ".format(gt_ids, processor.tokenizer.convert_ids_to_tokens(gt_ids)))

#     if not cb_ids:
#         raise ValueError(f"No valid token ids for cb_num={cb_num}, forms={cb_forms}")
#     if not gt_ids:
#         raise ValueError(f"No valid token ids for gt_num={gt_num}, forms={gt_forms}")
    
#     # Gather probs for each form, then take elementwise max
#     gt_probs = scores[:, gt_ids]         # [seq_len, len(gt_ids)]
#     cb_probs = scores[:, cb_ids]         # [seq_len, len(cb_ids)]
    
#     gt_max = gt_probs.max(dim=1).values  # [seq_len]
    
#     if is_negated:
#         print("IS NEGATED == TRUE")
#         # index of best GT token per timestep
#         gt_argmax = gt_probs.argmax(dim=1)   # [seq_len]
        
#         # select corresponding CB probability per timestep
#         cb_max = cb_probs.gather(
#             dim=1,
#             index=gt_argmax.unsqueeze(1)
#         ).squeeze(1)  

#         print("CB MAX SHAPE: ", cb_max.shape)
#     else:
#         cb_max = cb_probs.max(dim=1).values   

#     print(f"PROB OF {processor.tokenizer.convert_ids_to_tokens(cb_ids)}: {cb_max.max(axis=0)}")
#     print(f"PROB OF {processor.tokenizer.convert_ids_to_tokens(gt_ids)}: {gt_max.max(axis=0)}")
        
#     # Stack into [seq_len, 2]
#     selected = torch.stack([cb_max, gt_max], dim=1)

#     return torch.max(selected, axis=0).values.detach().cpu().numpy()


def process_results(results, cb_num, gt_num, processor, task='color', is_negated=False):
    if task == 'color':
        gt_num = ' ' + gt_num
        cb_num = ' ' + cb_num
    
    print("CB NUM: ", cb_num)
    print("GT NUM: ", gt_num)
    
    # STACK scores then get PROBABILITIES
    scores = torch.stack(results[0]["scores"])                # [seq_len, 1, vocab]
    scores = torch.stack([torch.softmax(s, dim=-1) for s in scores], dim=0)  
    scores = scores.squeeze(1)                                # -> [seq_len, vocab]
    vocab_size = scores.size(-1)
    
    def tokenize_to_ids(num_str):
        """Tokenize and return all valid token ids (skip specials)."""
        ids = processor.tokenizer(num_str, add_special_tokens=False).input_ids
        return [i for i in ids if i < vocab_size]
    
    def expand_forms(num):
        """Expand number into digit and word forms (always <=19)."""
        n = int(num)
        return {
            'digit': ' ' + str(n),
            'word': ' ' + num2words(n)
        }
    
    if task == 'count':
        # Expand cb and gt into digit+word forms
        cb_forms = expand_forms(cb_num)
        gt_forms = expand_forms(gt_num)
    elif task == 'color':
        # For color task, just use single form
        cb_forms = {'single': cb_num}
        gt_forms = {'single': gt_num}
    
    # Convert forms to token IDs (keep separate by form type)
    cb_ids_by_form = {form_type: tokenize_to_ids(form_str) 
                      for form_type, form_str in cb_forms.items()}
    gt_ids_by_form = {form_type: tokenize_to_ids(form_str) 
                      for form_type, form_str in gt_forms.items()}
    
    # Remove overlapping tokens for each form type separately
    cb_ids_unique = {}
    gt_ids_unique = {}
    
    for form_type in cb_forms.keys():
        cb_counter = Counter(cb_ids_by_form[form_type])
        gt_counter = Counter(gt_ids_by_form[form_type])
        
        cb_ids_unique[form_type] = list((cb_counter - gt_counter).elements())
        gt_ids_unique[form_type] = list((gt_counter - cb_counter).elements())
    
    # Calculate probabilities for each form separately
    results_dict = {}
    
    for form_type in cb_forms.keys():
        cb_ids = cb_ids_unique[form_type]
        gt_ids = gt_ids_unique[form_type]
        
        if not cb_ids:
            print(f"Warning: No valid token ids for cb_num={cb_num}, form={form_type}")
            cb_max = torch.zeros(scores.size(0))
        else:
            cb_probs = scores[:, cb_ids]  # [seq_len, len(cb_ids)]
            if is_negated:
                # For negated case, need to align with GT
                if not gt_ids:
                    cb_max = torch.zeros(scores.size(0))
                else:
                    gt_probs = scores[:, gt_ids]
                    gt_argmax = gt_probs.argmax(dim=1)
                    cb_max = cb_probs.gather(dim=1, index=gt_argmax.unsqueeze(1)).squeeze(1)
            else:
                cb_max = cb_probs.max(dim=1).values
        
        if not gt_ids:
            print(f"Warning: No valid token ids for gt_num={gt_num}, form={form_type}")
            gt_max = torch.zeros(scores.size(0))
        else:
            gt_probs = scores[:, gt_ids]  # [seq_len, len(gt_ids)]
            gt_max = gt_probs.max(dim=1).values
        
        # Store max probability across sequence for this form type
        results_dict[form_type] = {
            'cb_prob': cb_max.max().item(),
            'gt_prob': gt_max.max().item(),
            'cb_tokens': processor.tokenizer.convert_ids_to_tokens(cb_ids) if cb_ids else [],
            'gt_tokens': processor.tokenizer.convert_ids_to_tokens(gt_ids) if gt_ids else []
        }
        
        print(f"Form: {form_type}")
        print(f"  PROB OF {results_dict[form_type]['cb_tokens']}: {results_dict[form_type]['cb_prob']}")
        print(f"  PROB OF {results_dict[form_type]['gt_tokens']}: {results_dict[form_type]['gt_prob']}")

    
    # Return structured results
    return results_dict


def generate_with_probabilities_pih(model, model_version, processor, image_path, row, task):
    # Probably want to make this a dictionary with a idx key. 
    # want to process the different number conditions- Gt, CB, changed number 

    # Store confidence of generating number! 
    tokenizer=processor.tokenizer


    with torch.inference_mode():
        # REMOVE -- CLASS HERE? 
        confidence_by_class = {'class': row['class_label']}
        # IMAGE PATH for the ROW. 
        # Nested loop for each distance from the GT. 

        if task == 'count':
            column_set = ['prompt_count', 'clean_prompt', '1','2','3','4','5']
        
        elif task == 'color':
            column_set = ['prompt_count', 'clean_prompt', '1','2','3']
            
        for num in column_set:
            confidence_by_class[f'cb_{num}'] = {}
            results = []
            
            if num not in ["prompt_count", "clean_prompt"]:
                prompt = row[f'prompt_describe_plus_{num}']            
            
            if num == 'prompt_count':
                prompt = row[num]    
            
            if num == 'clean_prompt':
                if task == 'count':
                    prompt = row[f'prompt_describe_plus_1']    
                    new_number = row['number']
                    prompt = replace_number(prompt, new_number)
                elif task == 'color':
                    prompt = row[num]
            
            inputs = build_inputs(image_path, model_version, processor, prompt, model, task=task, device="cuda")
        
            if model_version == "janus":           
                 # removing 'generate' for now. 
                 outputs = model.language_model.generate(
                    inputs_embeds=inputs["inputs_embeds"],
                    attention_mask=inputs['attention_mask'],
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=75,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                    temperature=1.0,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            elif model_version == 'llava-one':   
                outputs = model.generate(**inputs, max_new_tokens=75, num_beams=1, do_sample=False, temperature=1.0, output_scores=True, return_dict_in_generate=True) 
                                           
            elif model_version == 'qwen':
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=75,
                    do_sample=False,
                    use_cache=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Decode to text
            generated_texts = processor.tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            print("GENERATED TEXT: ", generated_texts)

            results.append({"prompt": prompt, "answer": outputs.sequences, "scores": outputs.scores}) 
            torch.cuda.empty_cache()
                        
            print("ROW KEYS ", row.keys()) 
            #print('RESPONSE', processor.tokenizer.decode(outputs.sequences[0], add_special_tokens=False))
            
            if task == 'color':
                for curr_num in [0, 1, 2]:
                    gt_color = row['gt_color']

                    print(row[f'cb_class_{curr_num}'])
                    if row[f'cb_class_{curr_num}'] == 'GT-Negation':
                        is_negated = True 
                    else:
                        is_negated = False
    
                    cb_color = color_distances[gt_color][curr_num] 
                    num_probs = process_results(results, cb_color, gt_color, processor, task=task, is_negated=is_negated)
                    confidence_by_class[f'cb_{num}'][curr_num] = num_probs[0]
                    
            if task == 'count':
                for curr_num in ['1','2','3','4','5']:
                    # Get probabilities for both digit and word forms
                    num_probs = process_results(results, str(int(row['number']) + int(curr_num)), row['number'], processor, task=task)
                    
                    # Store probabilities for both digit and word forms
                    # CB probabilities
                    confidence_by_class[f'cb_{num}'][f'{curr_num}_digit'] = num_probs['digit']['cb_prob']
                    confidence_by_class[f'cb_{num}'][f'{curr_num}_word'] = num_probs['word']['cb_prob']
                    
                    # GT probabilities (same for all curr_num iterations, but we'll store on last iteration)
                    if curr_num == '5':  # Store GT probs once after loop
                        confidence_by_class[f'cb_{num}']['ground_truth_digit'] = num_probs['digit']['gt_prob']
                        confidence_by_class[f'cb_{num}']['ground_truth_word'] = num_probs['word']['gt_prob']

    return confidence_by_class




###################################################################################################################
############################################     KNOCKOUTS     ####################################################
###################################################################################################################

class HeadMeanAblation:
    """
    Zero-out selected attention heads by slicing the pre-o_proj tensor.
    Works for Qwen2 (model.model.layers) and LLaVA-Next (model.language_model.model.layers).
    """
    def __init__(self, model, model_version="qwen", alpha=0.0):
        self.model = model
        self.model_version = model_version
        self.hooks = []
        self.alpha=alpha

        cfg = getattr(model, "config", None)

        if model_version == "qwen":
            self.num_layers = getattr(cfg, "num_hidden_layers", None)
            self.num_heads  = getattr(cfg, "num_attention_heads", None)
            hidden_size     = getattr(cfg, "hidden_size", None)

        elif model_version == "janus":  # LLaVA-Next (Mistral backbone)
            text_cfg = getattr(cfg, "language_config", None)
            self.num_layers = getattr(text_cfg, "num_hidden_layers", None)
            self.num_heads  = getattr(text_cfg, "num_attention_heads", None)
            hidden_size     = getattr(text_cfg, "hidden_size", None)

        else:  # LLaVA-Next (Mistral backbone)
            text_cfg = getattr(cfg, "text_config", None)
            self.num_layers = getattr(text_cfg, "num_hidden_layers", None)
            self.num_heads  = getattr(text_cfg, "num_attention_heads", None)
            hidden_size     = getattr(text_cfg, "hidden_size", None)

        if self.num_layers is None or self.num_heads is None or hidden_size is None:
            raise RuntimeError("Could not infer model dims from config.")

        if hidden_size % self.num_heads != 0:
            raise RuntimeError("hidden_size not divisible by num_heads.")

        self.head_dim = hidden_size // self.num_heads
        self.to_zero = set()

    
    def _make_hook(self, layer_idx):
        head_dim  = self.head_dim
        num_heads = self.num_heads
    
        def hook(module, inputs):
            x = inputs[0]  # [B, T, H]
            if not self.to_zero:
                return
    
            Hsz = x.shape[-1]
            assert Hsz == num_heads * head_dim, f"Unexpected hidden size: {Hsz}"
    
            x_out = x.clone()
    
            for (L, h) in self.to_zero:
                if L != layer_idx:
                    continue
                start = h * head_dim
                end   = start + head_dim
    
                # Compute mean over [B, T, head_dim] slice                
                head_slice = x[:, :, start:end]                  # shape: [B, T, head_dim]
                mean_vec = head_slice.mean(dim=(0, 1), keepdim=True)  # shape: [1, 1, head_dim]
                
                # ADD ALPHAS HERE. 
                # Broadcast and assign mean to that head
                x_out[:, :, start:end] = (1-self.alpha)*mean_vec + self.alpha*x_out[:,:,start:end]
    
            return (x_out,)
    
        return hook

    def enable(self):
        if self.model_version == "qwen":
            layers = self.model.model.layers

        else:
            layers = self.model.language_model.model.layers

        for i, layer in enumerate(layers):
            o_proj = layer.self_attn.o_proj
            h = o_proj.register_forward_pre_hook(self._make_hook(i), with_kwargs=False)
            self.hooks.append(h)

    def disable(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @contextmanager
    def knockout(self, pairs):
        old = self.to_zero.copy()
        try:
            self.to_zero = set(pairs)
            self.enable()
            yield
        finally:
            self.disable()
            self.to_zero = old

# MAKE SURE TO SPECIFY ALPHA
def run_head_knockout_sweep(model, model_version, processor, image_path, row, heads_to_ablate=None,
                            max_new_tokens=50, do_sample=False, temperature=0.0,
                            limit_layers=None, limit_heads=None, alpha=0.0, task='color'):
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
    
    zeroer = HeadMeanAblation(model, model_version, alpha)
    changes = []
    
    # ------------------------------
    # SWEEP HEADS
    # ------------------------------
    for L in layer_iter:
        for H in head_iter:
            if heads_to_ablate:
                with zeroer.knockout(heads_to_ablate):
                    results = generate_with_probabilities_pih(model, model_version, processor, image_path, row, task)
            else:    
                with zeroer.knockout({(L, H)}):
                    results = generate_with_probabilities_pih(model, model_version, processor, image_path, row, task)

    return results

def main():

    parser = argparse.ArgumentParser(description="Generate probabilities after knockout.")
    parser.add_argument('--model_version', type=str, choices=['qwen' ,'janus', 'llava-one'], required=True, help="Choose the model version.")
    parser.add_argument('--corrupt', type=str, default='False')
    parser.add_argument('--heads_to_ablate', type=str)
    parser.add_argument("--num_samples", type=str, default="20")
    parser.add_argument("--task", type=str, default="color")
    parser.add_argument("--alpha", type=float, default=0.0, required=False)
    args = parser.parse_args()
    
    model_version=args.model_version

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    if model_version == 'llava-one':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True
        )


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

    print(df.head(1).keys())

    
    if args.num_samples != 'all':
        df = df.head(int(args.num_samples))


    if args.heads_to_ablate:
        print(args.heads_to_ablate)
        heads_to_ablate = head_ablate_dict[args.model_version][args.heads_to_ablate]
        print(heads_to_ablate)
        
        confidence_by_class = {}
        for row in df.reset_index().itertuples(index=False):
            i = row.index
            print(i)
            image_path   =  row.path
            
            if args.task == 'count':
                ground_truth = int(row.number)
            
            elif args.task == 'color':
                ground_truth = row.gt_color
            
            print(ground_truth)
            
            row = row._asdict()
            results = run_head_knockout_sweep(
                model, args.model_version, processor, image_path,
                max_new_tokens=10, do_sample=False, temperature=0.0,
                limit_layers=[1], limit_heads=[1], heads_to_ablate=heads_to_ablate, row=row, alpha=args.alpha, task=args.task)
            
            confidence_by_class[i] = results

        if args.num_samples == 'all':   
            print("SAVING")
            file_path = f"{model_version}_{args.task}_confidence_by_class_knockout_{args.alpha}_decoupled.pkl"
            print(file_path)
            with open(file_path, "wb") as f:
                pickle.dump(confidence_by_class, f)

    else:
        confidence_by_class = {}
        for row in df.reset_index().itertuples(index=False):
            i = row.index
            print(i)
            print("ROW {}, LABEL:  {}".format(i, row.class_label))
            image_path   =  row.path
            if args.task == 'count':
                ground_truth = int(row.number)
            
            elif args.task == 'color':
                ground_truth = row.gt_color
            
            row = row._asdict()
            results = generate_with_probabilities_pih(model, model_version, processor, image_path, row, task=args.task)
            confidence_by_class[i] = results

        if args.num_samples == 'all':   
            print("SAVING")
            file_path = f"{model_version}_{args.task}_confidence_by_class_decoupled.pkl"
            print(file_path)
            with open(file_path, "wb") as f:
                pickle.dump(confidence_by_class, f)
            
            

    
if __name__ == "__main__":
    main()