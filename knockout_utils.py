from general_utils import *
import pandas as pd
import re
from contextlib import contextmanager
from word2number import w2n

head_ablate_dict = {
    'qwen': {
        'top1': {(0, 3), (11, 15), (0, 6)},
        'top2': {(0, 24), (0, 3), (11, 15), (0, 6)},
        'top3': {(10, 24), (15, 8), (0, 3), (14, 20), (0, 6), (14, 13), (11, 15), (11, 21), (0, 24), (12, 11), (9, 7)},
        
        'best3': {(0, 3), (11, 15), (0, 6)},
        'best5': {(15, 8), (0, 3), (0, 6), (11, 15), (0, 11)},
        'best10': {(0, 10), (15, 8), (0, 3), (0, 6), (7, 3), (8, 13), (14, 13), (11, 15), (0, 24), (0, 11)},        

    },
    'janus': {
        'top1': {(1, 7), (0, 20)},
        'top2': {(0, 20), (14, 27), (12, 9), (1, 7), (11, 18), (0, 24), (8, 11)},
        'top3': {(8, 24), (3, 24), (1, 12), (0, 20), (14, 27), (6, 5), (12, 9), (1, 7), (6, 10), (14, 28), (11, 18), (0, 24), (8, 11)},
        
        'best3': {(1, 7), (0, 20), (14, 27)},
        'best5': {(0, 20), (14, 27), (12, 9), (1, 7), (11, 18)},
        'best10': {(1, 7), (0, 20), (14, 27)}, 
    }, 
    'llava-one': { 
        'top1': {(0, 3), (11, 15), (0, 6)},
        'top2': {(0, 23), (0, 26), (0, 3), (0, 6), (11, 15), (0, 24)},
        'top3': {(0, 23), (9, 16), (0, 26), (0, 3), (0, 6), (0, 15), (11, 15), (0, 24)},
        
        'best3': {(0, 3), (0, 26), (0, 6)},
        'best5': {(0, 23), (0, 26), (0, 3), (0, 6), (11, 15)},
        'best10': {(0, 23), (0, 26), (0, 3), (0, 6), (17, 22), (0, 15), (11, 15), (14, 9), (0, 24), (0, 11)},
    }
}

def diff_strings(a, b):
    return "\n".join(ndiff(a.split(), b.split()))


class HeadMeanAblation:
    """
    Zero-out selected attention heads by slicing the pre-o_proj tensor.
    Works for Qwen2 (model.model.layers) and LLaVA-Next (model.language_model.model.layers).
    """
    def __init__(self, model, model_version="qwen"):
        self.model = model
        self.model_version = model_version
        self.hooks = []

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
    
                # Broadcast and assign mean to that head
                x_out[:, :, start:end] = mean_vec
    
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

def run_head_knockout_sweep(task, model, model_version, processor, image_path, prompt, response,
                            max_new_tokens=50, do_sample=False, temperature=0.0,
                            limit_layers=None, limit_heads=None, heads=None):
    """
    Returns: changes (list of dicts:
             {layer, head, ablated_text, changed: bool, diff (optional)})
    """
    baseline_text = response
    inputs = build_inputs(task, image_path, model_version, processor, prompt, model)

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
    for L in layer_iter:
        for H in head_iter:
            
            if heads:
                with zeroer.knockout(heads): 
                    ablated = generate_text(model,processor, inputs, model_version, max_new_tokens, do_sample, temperature)
            else:
                with zeroer.knockout({(L, H)}):
                    ablated = generate_text(model,processor, inputs, model_version, max_new_tokens, do_sample, temperature)
            changed = (ablated != baseline_text)
            entry = {"layer": L, "head": H, "ablated_text": ablated, "changed": changed}
            if changed:
                entry["diff"] = diff_strings(baseline_text, ablated)
            changes.append(entry)
            print(f"[L{L:02d} H{H:02d}] {'CHANGED' if changed else 'same'}")
    
    return changes

def clean_llava_response(text: str) -> str:
    """
    Removes the input prompt from an LLaVA-style response.
    Everything from 'user' up to and including 'assistant' is removed.
    """
    return re.sub(r"user\s*.*?assistant\s*\n", "", text, flags=re.DOTALL)

def extract_color_from_assistant(text, model_version):
    COLOR_VARIANTS = [
        'crimson','scarlet','auburn','maroon','rust','burgundy','wine','red',
        'beige','tan','khaki','chocolate','coffee','caramel','mocha','brown',
        'amber','apricot','peach','tangerine','coral','orange',
        'golden','mustard','lemon','canary','flax','butter','yellow',
        'lime','olive','mint','sage','emerald','forest','jade','chartreuse','green',
        'navy','sky','cobalt','azure','teal','turquoise','cerulean','blue',
        'violet','lavender','magenta','plum','orchid','amethyst','purple',
        'rose','fuchsia','salmon','blush','bubblegum','pink',
        'ebony','charcoal','onyx','black',
        'ivory','cream','snow','pearl','alabaster','white',
        'slate','ash','stone','dove','smoke','graphite','grey','gray',
        'gold','goldenrod','silver','platinum','metallic'
    ]

    # longest-first helps prefix matching (so "reddish" -> "red", not "re")
    _VARIANTS_SORTED = sorted(set(COLOR_VARIANTS), key=len, reverse=True)

    # canonical mapping
    _MAPPING = {
        **{k: 'red' for k in ['red','crimson','scarlet','auburn','maroon','rust','burgundy','wine']},
        **{k: 'brown' for k in ['brown','beige','tan','khaki','chocolate','coffee','caramel','mocha']},
        **{k: 'orange' for k in ['orange','amber','apricot','peach','tangerine','coral']},
        **{k: 'yellow' for k in ['yellow','golden','mustard','lemon','canary','flax','butter']},
        **{k: 'green' for k in ['green','lime','olive','mint','sage','emerald','forest','jade','chartreuse']},
        **{k: 'blue' for k in ['blue','navy','sky','cobalt','azure','teal','turquoise','cerulean']},
        **{k: 'purple' for k in ['purple','violet','lavender','magenta','plum','orchid','amethyst']},
        **{k: 'pink' for k in ['pink','rose','fuchsia','salmon','blush','bubblegum']},
        **{k: 'black' for k in ['black','ebony','charcoal','onyx']},
        **{k: 'white' for k in ['white','ivory','cream','snow','pearl','alabaster']},
        **{k: 'gray' for k in ['gray','grey','slate','ash','stone','dove','smoke','graphite']},
        'gold': 'gold', 'goldenrod': 'gold',
        'silver': 'silver', 'platinum': 'silver', 'metallic': 'silver'
    }
    

    if model_version == 'llava-one':
        text = clean_llava_response(text)
    
    out = []
    # get letter-or-hyphen chunks to preserve hyphen groups
    chunks = re.findall(r'[A-Za-z-]+', text)
    for chunk in chunks:
        for part in chunk.split('-'):
            if not part:
                continue
            p = part.lower()
            # find the longest variant that is a prefix of p (handles 'reddish' -> 'red')
            found = next((v for v in _VARIANTS_SORTED if p.startswith(v)), None)
            if not found:
                continue
            # map to canonical color (skip if no mapping)
            canon = _MAPPING.get(found)
            if canon:
                out.append(canon)
    return out


def extract_assistant_reply(text: str, model_version: str) -> str:
    _QWEN_ASSISTANT_LINE_RE = re.compile(r'(?mi)^\s*assistant\s*$', re.IGNORECASE)
    _L1_6_ASSISTANT_RE      = re.compile(r'\[/INST\]', re.IGNORECASE)
    # llava-one: try strict line-bounded first, then fallback to any word-boundary "assistant"
    _LLAVA_ONE_ASSISTANT_LINE_RE  = re.compile(r'(?mi)(?:^|\n)\s*assistant\s*(?:\n|$)')
    _LLAVA_ONE_ASSISTANT_WORD_RE  = re.compile(r'(?i)\bassistant\b')

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

def extract_number_from_assistant(text: str, model_version: str):
    """
    Wrapper: scopes extraction to the assistant completion
    (and optionally only after 'There are').
    """
    scope = extract_assistant_reply(text, model_version)
    return extract_number_first_token_or_word(scope)

def summarize_head_changes(task, model_version, baseline_text, changes, ground_truth=None, include_diff=False):
    if task == "color":
        base_val = extract_color_from_assistant(baseline_text, model_version)
    else:
        base_val = extract_number_from_assistant(baseline_text, model_version)

    rows = []
    for c in changes:
        layer = c["layer"]; head = c["head"]; ablated_text = c["ablated_text"]
        if task == "color":
            ablated_val = extract_color_from_assistant(ablated_text, model_version)
        else:
            ablated_val = extract_number_from_assistant(ablated_text, model_version)
        
        row = {
            "layer": layer, "head": head,
            "baseline_value": base_val, "ablated_value": ablated_val,
        }
        if include_diff and "diff" in c:
            row["diff"] = c["diff"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["layer","head"]).reset_index(drop=True)
    return df

def extract_entity_color(text):
    """
    Extracts text after 'describe the' from a string.
    Removes trailing periods and extra spaces.
    Returns None if no match.
    """
    if not isinstance(text, str):
        return None
    pattern = re.compile(r'describe the\s+(.*?)(?:\.\s*$|$)', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None

def categorize_row_color(row):
    ablated_text = str(row["ablated_text"])
    baseline_text = str(row["baseline_text"])

    # truncate baseline to ablated length
    #truncated_baseline = baseline_text[:len(ablated_text)]
    MAX_TOKENS = 50

    baseline_tokens = baseline_text.split()
    truncated_baseline = " ".join(baseline_tokens[:MAX_TOKENS])

    ablated = ablated_text.lower()
    truncated_baseline = truncated_baseline.lower()

    # equality check AFTER truncation
    if ablated_text == truncated_baseline:
        return "same"

    gt = str(row.get("gt_entity", "")).lower()
    cb = str(row.get("cb_entity", "")).lower()
    cb_color = str(row.get("cb", "")).lower().strip()
    gt_color = str(row.get("ground_truth", "")).lower().strip()

    gt_present = bool(re.search(rf"\b{re.escape(gt_color)}\b", ablated))
    cb_present = bool(re.search(rf"\b{re.escape(cb_color)}\b", ablated))


    if cb not in truncated_baseline:
        #blue yam example
        return "truncation_unclear"
        
    # Rule: CB entity appears in ablated text (and not negated)
    if cb in ablated and all(x not in ablated for x in [" no ", " not "]): # and all(x not in ablated for x in [" no ", " not "]):
        return "cb_entity_mentioned"

    # Rule 1: GT entity appears in ablated text
    if gt in ablated:
        return "gt_entity_mentioned"
        
    #saying "white" for a white arctic fox    
    if gt_present and not cb_present:
        return "gt_color_only"

    if cb_present and not gt_present:
        return "cb_color_only"

    # Rule 2: CB entity does not appear in ablated text
    #not saying "pink arctic fox"
    if cb not in ablated:
        return "not_cb"

    # Rule 3: fallback
    return "text_changed"