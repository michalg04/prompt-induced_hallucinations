from PIL import Image
from io import BytesIO
import torch
import pandas as pd
import re
import numpy as np
import random
from janus.utils.io import load_pil_images
import base64
import gc
import ast
from qwen_vl_utils import process_vision_info



_QWEN_ASSISTANT_LINE_RE = re.compile(r'(?mi)^\s*assistant\s*$', re.IGNORECASE)
_L1_6_ASSISTANT_RE      = re.compile(r'\[/INST\]', re.IGNORECASE)
_LLAVA_ONE_ASSISTANT_LINE_RE  = re.compile(r'(?mi)(?:^|\n)\s*assistant\s*(?:\n|$)')
_LLAVA_ONE_ASSISTANT_WORD_RE  = re.compile(r'(?i)\bassistant\b')


def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def clean_janus_answer(text):
    pattern = r"<\|Assistant\|>:\s?(.*?)(?=<｜end▁of▁sentence｜>)"
    match = re.search(pattern, text)
    return match.group(1).strip('.')

def clean_llava_answer(text: str) -> str:
    """
    Removes the input prompt from an LLaVA-style response.
    Everything from 'user' up to and including 'assistant' is removed.
    """
    return re.sub(r"user\s*.*?assistant\s*\n", "", text, flags=re.DOTALL)


def clean_qwen_answer(text: str):
    # Match <|im_start|>assistant or |im_start|>assistant
    pattern = r"[<]?\|im_start\|\>assistant\s*(.*?)(?=[<]?\|im_start\|\>|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None 

def clean_answer(text, model_version):
    if model_version=='qwen':
        text=clean_qwen_answer(text)
    if model_version=='llava-one':
        text=clean_llava_answer(text)
    if model_version=='janus':
        text=text#clean_janus_answer(text)
    return text

def resize_by_shortest_side(image, target_shortest=224):
    w, h = image.size
    short_side, long_side = min(w, h), max(w, h)

    # Compute scale so that shortest side becomes `target_shortest`
    scale = target_shortest / short_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    return image.resize((new_w, new_h), Image.LANCZOS)

def build_inputs(task, image_path, model_version, processor, prompt, model, device="cuda"):
    """
    Returns a dict suitable for the corresponding generate() call for each model_version.
    - qwen / 1.6 / llava-one: regular tokenized inputs for model.generate(**inputs)
    - janus: {'inputs_embeds': ..., 'attention_mask': ...} for model.language_model.generate(...)
    """
    if task == "count":
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.open(BytesIO(image_path)).convert("RGB")

    img = resize_by_shortest_side(img, target_shortest=224)

    if model_version == "qwen":
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img, 
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        dev = model.device if model is not None else device
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
       
        return inputs.to(dev)


    elif model_version == "llava-one":  
        conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type":"image"}
                    ],
                },
            ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
            ).to(model.device, torch.float16 if model.device.type == "cuda" else torch.float32)
        
        return inputs

    elif model_version == "janus":

        image_data = base64.b64encode(image_path).decode("utf-8")
        image = f"data:image/jpeg;base64,{image_data}" 
        
        conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
        ]
        
        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        
        prepare_inputs = processor(
        conversations=conversation, images=pil_images, force_batchify=True
        ).to(model.device)
        
        # # run image encoder to get the image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        # Return the exact mapping Janus' generate call needs
        return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": prepare_inputs.attention_mask,  # keep mask with same device/dtype
        }

    else:
        raise ValueError(f"Unknown model_version: {model_version}")
    

def generate_text(model, processor, inputs, model_version,
                  max_new_tokens=50, do_sample=False, temperature=0.0):
    """
    Calls the correct generate() for each model family and decodes the text.
    Deterministic by default (do_sample=False).
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        output_attentions=False,
        use_cache=True,
        num_beams=1,           # keep greedy
    )

    with torch.no_grad():
        if model_version == "janus":
            # Janus uses LM.generate with inputs_embeds + mask and Janus tokenizer to decode
            tok = processor.tokenizer
            out = model.language_model.generate(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=tok.eos_token_id,
                bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id,
                **gen_kwargs,
            )
            text = tok.decode(out[0].tolist(), skip_special_tokens=True)
        
        elif model_version == "qwen":
            generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            text = output_text[0]

        else:
            # Qwen / LLaVA variants follow the standard path
            out = model.generate(**inputs, **gen_kwargs)
            text = processor.batch_decode(out, skip_special_tokens=True)[0]

    return text


def eval_color_pih(df, processor, model, model_name, task):

    color_distances = {
        # Rainbow + Brown cycle
        'red':      ['purple', 'blue', 'green'], 
        'orange':   ['red', 'purple', 'blue'],
        'yellow':   ['orange', 'red', 'purple'],
        'green':    ['yellow', 'orange', 'red'],
        'blue':     ['green', 'yellow', 'orange'],
        'purple':   ['blue', 'green', 'yellow']}  

    # Store texts for each of the color distances
    generated_texts = {idx:[] for idx in range(3)} 
    prompts = {idx:[] for idx in range(3)} 
    
    with torch.inference_mode():
        torch.cuda.empty_cache()
        gc.collect()
        for idx, row in df.iterrows():
            if idx%10==0:
                print(f"{idx} ROWS PROCESSED")

            image_path = row['original_image']['bytes']    
            object_name = row['object']
            for color_distance in range(3):
                # make prompt.
                corrupt_color = color_distances[ast.literal_eval(row['correct_answer'])[0]][color_distance]
                question = f"Describe the {corrupt_color} {object_name}."
                 
                inputs = build_inputs(task, image_path, model_name, processor, question, model, device="cuda")
                predicted_answer = generate_text(model, processor, inputs, model_name, max_new_tokens=50, do_sample=False, temperature=0.0)
                
                generated_texts[color_distance].append(predicted_answer)
                prompts[color_distance].append(question)
                
        for color_distance in range(3):
            df[f'response_prompt_describe_plus_{color_distance}'] = generated_texts[color_distance]
            df[f'prompt_describe_plus_{color_distance}'] = prompts[color_distance]
    
        if 'inputs' in locals(): del inputs
        if 'image' in locals(): del image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    return df


def eval_color_base(df, processor, model, model_name, task):
    # Store texts for each of the color distances
    generated_texts = {}
    prompts = {}
    
    with torch.inference_mode():
        torch.cuda.empty_cache()
        gc.collect()
        for idx, row in df.iterrows():
            if idx%10==0:
                print(f"{idx} ROWS PROCESSED")

            image_path = row['original_image']['bytes']    
            object_name = row['object']
           
            # make prompt.
            question = f"What color is the {object_name} in the image?"
             
            inputs = build_inputs(task, image_path, model_name, processor, question, model, device="cuda")
            predicted_answer = generate_text(model, processor, inputs, model_name, max_new_tokens=50, do_sample=False, temperature=0.0)
                
            generated_texts[idx] = predicted_answer
            prompts[idx] = question
            print(predicted_answer)

        df['prompt_base'] = df.index.map(prompts.get)
        df['response_prompt_base'] = df.index.map(generated_texts.get)
            
    return df


def eval_counts(df, processor, model, model_name, task):

    for col in ["prompt_base"] + [f"prompt_describe_plus_{i}" for i in [1, 2, 3, 4, 5, 10, 20, 50]]:
        pred_col = f"response_{col}"
        df[pred_col] = ""
        
    for index, row in df.iterrows():
        SEED = 42
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.empty_cache()

        for col in ["prompt_base"] + [f"prompt_describe_plus_{i}" for i in [1, 2, 3, 4, 5, 10, 20, 50]]:
            question = row[col]
            image_path = row['path']
            inputs = build_inputs(task, image_path, model_name, processor, question, model, device="cuda")
            predicted_answer = generate_text(model, processor, inputs, model_name, max_new_tokens=50, do_sample=False, temperature=0.0)
                
    
            df.at[index, f"response_{col}"] = predicted_answer
    
    return df


def make_vertical_df(df):
    prompt_cols   = [c for c in df.columns if c.startswith("prompt_describe_plus_")]
    response_cols = [c for c in df.columns if c.startswith("response_prompt_describe_plus_")]

    # everything else is an id column
    id_cols = df.columns.difference(prompt_cols + response_cols).tolist()

    # melt
    p = df.melt(id_vars=id_cols, value_vars=prompt_cols,
                var_name="var", value_name="prompt")
    r = df.melt(id_vars=id_cols, value_vars=response_cols,
                var_name="var", value_name="response")

    # extract i = 1,2,5,10,...
    p["i"] = p["var"].str.extract(r"(\d+)$").astype(int)
    r["i"] = r["var"].str.extract(r"(\d+)$").astype(int)

    # align prompt_i with response_i
    out = (
        p.merge(r.drop(columns="var"), on=id_cols + ["i"], how="left")
        .drop(columns="var")
        .sort_values(id_cols + ["i"])
        .reset_index(drop=True)
    )

    # keep only real prompt rows
    out = out[out["prompt"].notna()]
    return out


def extract_number_first_token_or_word(text: str):
    if not isinstance(text, str):
        return None

    # Tokenize both digits and words in order
    tokens = re.findall(r"\b\d+\b|[A-Za-z]+(?:-[A-Za-z]+)*", text)

    for tok in tokens:
        low = tok.lower()
        if low in {"a", "an"}:
            continue
        if tok.isdigit():
            return int(tok)
        try:
            return w2n.word_to_num(low)
        except Exception:
            continue
    return None
    
def extract_number_from_assistant(text: str, model_version: str, use_anchor: bool = False):
    """
    Wrapper: scopes extraction to the assistant completion
    (and optionally only after 'There are').
    """
    scope = extract_after_anchor(text, model_version) if use_anchor \
            else extract_assistant_reply(text, model_version)
    return extract_number_first_token_or_word(scope)

def extract_assistant_reply(text: str, model_version: str) -> str:
    if not isinstance(text, str):
        return ""

    if model_version == "qwen":
        last = None
        for m in _QWEN_ASSISTANT_LINE_RE.finditer(text):
            last = m
        return text[last.end():].lstrip("\n") if last else text

    elif model_version == "1.6":
        last = None
        for m in _L1_6_ASSISTANT_RE.finditer(text):
            last = m
        return text[last.end():].lstrip() if last else text

    elif model_version == "llava-one":
        # 1) strict (assistant on its own line)
        last = None
        for m in _LLAVA_ONE_ASSISTANT_LINE_RE.finditer(text):
            last = m
        if last is None:
            # 2) fallback: any word-boundary "assistant"
            for m in _LLAVA_ONE_ASSISTANT_WORD_RE.finditer(text):
                last = m
        return text[last.end():].lstrip() if last else text

    return text

def categorize_change_with_nums(baseline_text, ablated_text, model_version, ground_truth=None, use_anchor=False):
    print(ablated_text)
    print(baseline_text)
    anum = extract_number_from_assistant(ablated_text, model_version, use_anchor=use_anchor)
    
    print(anum)
    bnum = extract_number_from_assistant(baseline_text, model_version, use_anchor=use_anchor)
    print(bnum)
    if ablated_text == baseline_text:
        return "same"

    if bnum == anum:
        return "text changed, number stayed the same"

    if ground_truth is not None and anum == ground_truth:
        return "number changed to ground truth"

    return "number changed (≠ ground truth)"

def collect_top_heads_with_ties(df, k):
    """
    For each ground_truth:
    - rank heads by percent (ties get same rank)
    - keep all heads with rank <= k
    - return union set of (layer, head)
    """

    # Rank within each ground_truth (ties share rank)
    df["rank"] = (
        df.groupby("ground_truth")["percent"]
        .rank(method="dense", ascending=False)
    )

    topk = df[df["rank"] <= k]

    return set(zip(topk["layer"].astype(int), topk["head"].astype(int)))

def replace_number(prompt: str, new_number: int) -> str:
    return re.sub(r'\b\d+\b', str(new_number), prompt)


def extract_color(text):
    COLORS = ['red','brown','orange','yellow','green','blue','purple','pink',
              'black','white','gray','grey','gold','silver']
    pattern = re.compile(r'\b(' + '|'.join(COLORS) + r')\b', flags=re.IGNORECASE)
    m = pattern.search(text)
    return ' ' + m.group(0) if m else None

color_distances = {
    'red':      ['purple', 'blue', 'green'], 
    'orange':   ['red', 'purple', 'blue'],
    'yellow':   ['orange', 'red', 'purple'],
    'green':    ['yellow', 'orange', 'red'],
    'blue':     ['green', 'yellow', 'orange'],
    'purple':   ['blue', 'green', 'yellow'],
}


def extract_colors(text):
    """Extract clean lowercase words from text (strips punctuation)."""
    return re.findall(r"[a-zA-Z]+", text.lower())

def check_correct_color(row):
    # Parse list
    correct_colors = ast.literal_eval(row["correct_answer"])
    correct_colors = [c.lower() for c in correct_colors]

    # Clean response words
    response_words = extract_colors(row["response_prompt_color"])

    # Match
    return any(color in response_words for color in correct_colors)



def pluralize_last_token(entity: str) -> list:
    """Return possible fuzzy versions of an entity: singular core + plural core."""
    if not entity:
        return []
    tokens = entity.split()
    core = tokens[-1]
    plural = p.plural(core)

    fuzzy = []
    fuzzy.append(core.lower())
    if plural.lower() != core.lower():
        fuzzy.append(plural.lower())

    return fuzzy


NEGATION_PATTERNS = [
    r"not", r"no", r"never", r"without", r"none", r"n't",
    r"no such thing", r"cannot see", r"can't see",
    r"does not contain", r"doesn't contain",
    r"does not have", r"doesn't have",
    r"is not", r"isn't",
    r"not present", r"absent",
    r"cannot detect", r"unable to see",
    r"there is no existence of"
]


def has_negation_of_color_entity(text, color, entity, fuzzy_entities):
    """
    Check if the response negates 'color + entity' or 'color + fuzzy-entity'.
    More robust:
      - allows newlines between tokens
      - allows negation before COLOR then ENTITY OR before ENTITY then COLOR
    """
    text = text.lower()
    color_esc = re.escape(color.lower())

    # Build regex for full entity + fuzzy variants
    entity_variants = [entity.lower()] + fuzzy_entities
    entity_variants = [e for e in entity_variants if e]  # filter empties
    entity_variants_esc = [re.escape(e) for e in entity_variants]
    entity_group = r"(?:%s)" % "|".join(entity_variants_esc)

    # Expanded / slightly looser negation phrases
    neg_list = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b", r"\bnone\b", r"n't\b",
        r"\bno such thing(?:\s+as)?\b",
        r"\bno existence of\b",
        r"\bcannot see\b", r"\bcan't see\b",
        r"\bdoes not contain\b", r"\bdoesn't contain\b",
        r"\bdoes not have\b", r"\bdoesn't have\b",
        r"\bis not\b", r"\bisn't\b",
        r"\bnot present\b", r"\babsent\b",
        r"\bcannot detect\b", r"\bunable to see\b",
    ]

    neg = r"(?:%s)" % "|".join(neg_list)

    # Case 1: neg ... color ... entity
    pattern1 = rf"{neg}.*?\b{color_esc}\b.*?\b{entity_group}\b"
    # Case 2: neg ... entity ... color
    pattern2 = rf"{neg}.*?\b{entity_group}\b.*?\b{color_esc}\b"

    flags = re.IGNORECASE | re.DOTALL

    if re.search(pattern1, text, flags=flags) or re.search(pattern2, text, flags=flags):
        return True
    return False



def first_color_entity_mention(text, colors, entity, fuzzy_entities):
    """
    Return the first match in the text among:
      - color + entity
      - color + fuzzy entity
    Returns tuple: (color, entity_type, position)
        entity_type = 'full' or 'fuzzy'
    If none found: return None
    """
    matches = []

    entity_esc = re.escape(entity.lower())
    fuzzy_esc = [re.escape(f) for f in fuzzy_entities]

    for color in colors:
        color_esc = re.escape(color)

        # Full entity match
        pat_full = rf"\b{color_esc}\b.*?\b{entity_esc}\b"
        for m in re.finditer(pat_full, text, flags=re.IGNORECASE):
            matches.append((color, "full", m.start()))

        # Fuzzy entity match
        for fe in fuzzy_esc:
            pat_fuzzy = rf"\b{color_esc}\b.*?\b{fe}\b"
            for m in re.finditer(pat_fuzzy, text, flags=re.IGNORECASE):
                matches.append((color, "fuzzy", m.start()))

    if not matches:
        return None

    return sorted(matches, key=lambda x: x[2])[0]  # earliest match


def classify_row(text, entity, gt_colors, cb_color):
    text_l = text.lower()
    fuzzy_entities = pluralize_last_token(entity.lower())

    # --- RULE 0: cb_color in GT → NaN ---
    if cb_color in [c.lower() for c in gt_colors]:
        return "NaN"

    # --- RULE 1: Negation of ANY relevant color+entity → GT-Negation ---
    # Check CB color
    if has_negation_of_color_entity(text_l, cb_color, entity.lower(), fuzzy_entities):
        return "GT-Negation"

    # Check ALL GT colors
    for gt_col in gt_colors:
        if has_negation_of_color_entity(text_l, gt_col, entity.lower(), fuzzy_entities):
            return "GT-Negation"

    # --- RULE 2: First mention wins ---
    all_colors = set(gt_colors + [cb_color])
    first = first_color_entity_mention(text_l, all_colors, entity.lower(), fuzzy_entities)

    if first is None:
        return "GT"

    color_found, _, _ = first

    if color_found == cb_color:
        return "CB"
    else:
        return "GT"
    

def process(df, color_distances):
    """
    Adds three classification columns to df:
        cb_class_0, cb_class_1, cb_class_2
    """
    # Prepare empty lists
    out0, out1, out2 = [], [], []

    for _, row in df.iterrows():
        entity = row["object"]
        gt_colors = [c.lower() for c in ast.literal_eval(row["correct_answer"])]
        gt_single = gt_colors[0]  # your rule
        
        # -------- For each of the 3 CB attempts -------- #

        row_classes = []

        for dist_idx in range(3):
            response_text = row[f"color_distance_{dist_idx}"]
            cb_color = color_distances[gt_single][dist_idx].lower()

            classification = classify_row(
                text=response_text,
                entity=entity,
                gt_colors=gt_colors,
                cb_color=cb_color
            )
            row_classes.append(classification)

        # Append results
        out0.append(row_classes[0])
        out1.append(row_classes[1])
        out2.append(row_classes[2])

    # Add new columns to DF
    df["cb_class_0"] = out0
    df["cb_class_1"] = out1
    df["cb_class_2"] = out2

    return df



def normalize_classes(seq):
    """
    seq = [cb_class_0, cb_class_1, cb_class_2]
    Normalize GT-negation → GT
    Remove NaN
    """
    clean = []
    for x in seq:
        if x is None:
            continue
        if isinstance(x, float) and pd.isna(x):
            continue
        if x == "NaN":
            continue

        if x == "GT-Negation":
            clean.append("GT")
        else:
            clean.append(x)
    return clean


def classify_final_row(is_correct, seq):
    """
    seq is the cleaned list of str like ["CB", "GT", ...]
    """

    # Category D: incorrect
    if not is_correct:
        return "D"

    # If everything was NaN → treat as C (all GT)
    if len(seq) == 0:
        return "C"

    # All CB
    if all(x == "CB" for x in seq):
        return "B"

    # All GT
    if all(x == "GT" for x in seq):
        return "C"

    # Category A: starts with CB
    if seq[0] == "CB":
        # A1: CB → GT → GT
        if all(x == "GT" for x in seq[1:]):
            return "A"
        # A2: CB → CB → GT
        if len(seq) > 1 and seq[1] == "CB" and (len(seq)==2 or seq[2]=="GT"):
            return "A"
        # More general: first is CB, rest not containing flip GT→CB
        if "GT" in seq[1:] and "CB" not in seq[1:]:
            return "A"
        if seq.count("CB") >= 1 and seq[-1] == "GT":
            return "A"

    # Mixed: starts with GT and flips to CB
    if seq[0] == "GT" and "CB" in seq[1:]:
        return "Mixed"

    # fallback —
    # if we get here, treat as C (all GT-like)  
    return "C"



def add_final_class(df):
    out = []
    for _, row in df.iterrows():
        seq = [
            row["cb_class_0"],
            row["cb_class_1"],
            row["cb_class_2"]
        ]
        clean_seq = normalize_classes(seq)

        label = classify_final_row(
            is_correct=row["is_correct_color"],
            seq=clean_seq
        )
        out.append(label)

    df["class_label"] = out
    return df

def process_color_df(df, model_name):
    COLORS = ['red','brown','orange','yellow','green','blue','purple','pink',
              'black','white','gray','grey','gold','silver']
    
    pattern = re.compile(r'\b(' + '|'.join(COLORS) + r')\b', flags=re.IGNORECASE)
    df["gt_color"] = df["correct_answer"].apply(lambda x: ast.literal_eval(x)[0])
    
    
    color_distances = {
        # Rainbow + Brown cycle
        'red':      ['purple', 'blue', 'green'], 
        'orange':   ['red', 'purple', 'blue'],
        'yellow':   ['orange', 'red', 'purple'],
        'green':    ['yellow', 'orange', 'red'],
        'blue':     ['green', 'yellow', 'orange'],
        'purple':   ['blue', 'green', 'yellow']} 
    
    
    prompts = {0:[], 1:[], 2:[]}
    clean_prompt = []
    prompt_count = []
    
    for idx, row in df.iterrows():
        for color_distance in range(3):
            object_name = row['object']
            # make prompt.
            corrupt_color = color_distances[row['gt_color']][color_distance]
            question = f"Describe the {corrupt_color} {object_name}."
            prompts[color_distance].append(question)
        
        clean_color = row['gt_color']
        question = f"Describe the {clean_color} {object_name}."
        clean_prompt.append(question)
    
        question = f"What color is the {object_name} in the image?"
        prompt_count.append(question)
    
    df['clean_prompt'] = clean_prompt
    df['prompt_count'] = prompt_count
    
    for idx in prompts.keys():
        df[f'prompt_describe_plus_{idx+1}'] = prompts[idx]


    df["is_correct_color"] = df.apply(check_correct_color, axis=1)

    if model_name == "llava-one": 
        cols = ["color_distance_0", "color_distance_1", "color_distance_2", "response_prompt_color"]
        df[cols] = df[cols].apply(lambda col: col.str.split(".assistant\n").str[1])
    # Create empty columns first (optional but clean)
    for i in range(3):
        df[f"cb_prompt_{i}"] = None
    
    # Fill prompts
    for idx, row in df.iterrows():
    
        object_name = row["object"]
        gt_single = ast.literal_eval(row["correct_answer"])[0]  # first GT color
    
        for color_distance in range(3):
            corrupt_color = color_distances[gt_single][color_distance]
            question = f"Describe the {corrupt_color} {object_name}."
        df.at[idx, f"cb_prompt_{color_distance}"] = question

    df = process(df, color_distances)
    df = add_final_class(df)
    
    return df