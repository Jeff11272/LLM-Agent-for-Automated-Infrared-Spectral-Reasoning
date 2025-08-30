import json
import re
import ast
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

def _normalize_label(s: str) -> str:
    if s is None:
        return ""
    # Unify full-width/strange quotes, remove surrounding quotes and extra punctuation
    s = s.strip().strip("'\"“”‘’`").strip()
    s = re.sub(r"\s+", " ", s)  # Merge whitespace
    return s

def _best_label_match(raw: str, allowed: Optional[List[str]], synonyms: Optional[Dict[str, str]]) -> str:
    x = _normalize_label(raw).lower()
    if synonyms:
        # Synonym/alias mapping (keys should be lowercase)
        if x in synonyms:
            x = synonyms[x].lower()
    if not allowed:
        return x
    # Case-insensitive closest match in allowed set (exact match prioritized)
    lower_map = {a.lower(): a for a in allowed}
    if x in lower_map:
        return lower_map[x]
    # Allow matching without spaces
    x_nospace = x.replace(" ", "")
    for a in allowed:
        if a.lower().replace(" ", "") == x_nospace:
            return a
    # If no match found, return original (or UNKNOWN)
    return lower_map.get(x, raw)

def _extract_code_blocks(text: str) -> List[str]:
    # Extract ```xxx``` code block content
    blocks = re.findall(r"```(?:json|python|txt|yaml|md)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return [b.strip() for b in blocks if b.strip()]

def _try_parse_list_like(s: str):
    """
    Try to parse string into list (JSON/py-list), return None on failure
    """
    # Try JSON (including embedded array fragments)
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        # Some models output {"labels":[...]}
        if isinstance(obj, dict):
            for k in ("labels", "prediction", "predictions", "y", "output"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
    except Exception:
        pass
    # Try to extract first JSON array fragment from text
    for m in re.finditer(r"\[[^\[\]]*?\]", s, flags=re.DOTALL):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if isinstance(obj, list):
                return obj
        except Exception:
            continue
    # Try python literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # Might be {index: label} format
            # Return value list sorted by keys
            try:
                keys = sorted(obj.keys(), key=lambda x: int(x))
            except Exception:
                keys = list(obj.keys())
            return [obj[k] for k in keys]
    except Exception:
        pass
    return None

def _parse_lines(s: str) -> List[str]:
    lines = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue
        # Handle "1. label" / "1) label" / "- label" / "* label"
        m = re.match(r"^\s*(?:[-*]\s+|(\d+)[\.\)\-:]\s*)(.+)$", t)
        if m:
            label = m.group(2) if m.group(1) else t.lstrip("-* ").strip()
            lines.append(label)
            continue
        # Handle "index: label"
        m2 = re.match(r"^\s*([^:]+)\s*:\s*(.+)$", t)
        if m2:
            lines.append(m2.group(2))
            continue
        # Regular line
        lines.append(t)
    return lines

def robust_parse_predictions(
    text: str,
    n_expected: Optional[int] = None,
    allowed_labels: Optional[List[str]] = None,
    label_synonyms: Optional[Dict[str, str]] = None,
    numeric_to_label: Optional[List[str]] = None,
) -> List[str]:
    """
    Try to parse model output into label list as much as possible.
    - n_expected: Expected length, for truncation/padding
    - allowed_labels: Allowed label set (for normalization mapping)
    - label_synonyms: Synonym mapping (lowercase -> canonical label)
    - numeric_to_label: If model outputs numbers/indices, map using this table
    """
    if not text or not text.strip():
        return []

    # 1) First try to parse code blocks
    blocks = _extract_code_blocks(text)
    for b in blocks:
        arr = _try_parse_list_like(b)
        if isinstance(arr, list) and arr:
            raw_list = arr
            break
    else:
        # 2) Try to parse full text directly
        arr = _try_parse_list_like(text)
        if isinstance(arr, list) and arr:
            raw_list = arr
        else:
            # 3) Fallback: split by comma or newline
            # First check if it's a comma-separated "flat" list
            if "," in text and ("\n" not in text or text.index(",") < text.index("\n")):
                parts = [p.strip() for p in text.split(",") if p.strip()]
            else:
                parts = _parse_lines(text)
            raw_list = parts

    # 4) Clean: remove surrounding quotes, whitespace
    cleaned = [_normalize_label(str(x)) for x in raw_list]

    # 5) If all are numbers/convertible to numbers, map to category indices
    if cleaned and all(re.fullmatch(r"[+-]?\d+", c) for c in cleaned):
        if numeric_to_label:
            mapped = []
            for c in cleaned:
                idx = int(c)
                if 0 <= idx < len(numeric_to_label):
                    mapped.append(numeric_to_label[idx])
                else:
                    mapped.append("UNKNOWN")
            cleaned = mapped
        # Otherwise keep numeric strings, but allowed_labels will normalize later

    # 6) Normalize to allowed_labels / synonyms
    final = []
    for c in cleaned:
        final.append(_best_label_match(c, allowed_labels, label_synonyms))

    # 7) Length alignment
    if n_expected is not None:
        if len(final) > n_expected:
            final = final[:n_expected]
        elif len(final) < n_expected:
            final = final + ["UNKNOWN"] * (n_expected - len(final))

    return final


class SpectrumCLS:
    def __init__(
        self,
        train_data: list,
        test_data: list,
        true_labels_test: list,
        api_key,
        base_url,
        model,
        temperature: float = 0.5,
        few_shot_initial: int = None,
        allowed_labels: Optional[List[str]] = None,
        label_synonyms: Optional[Dict[str, str]] = None,
        numeric_label_order: Optional[List[str]] = None,  # e.g. ["A","B","C"] for 0/1/2 mapping
        use_json_mode: bool = False,  # If backend supports, can try JSON mode
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.true_labels_test = true_labels_test

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        self.TEMPERATURE = temperature
        self.FEW_SHOT_INITIAL = few_shot_initial

        # If allowed_labels not explicitly provided, infer from train/test ground truth
        if allowed_labels is None:
            labels_set = set()
            for e in (train_data or []):
                if "label" in e:
                    labels_set.add(str(e["label"]))
            for t in (true_labels_test or []):
                labels_set.add(str(t))
            allowed_labels = sorted(labels_set)
        self.allowed_labels = allowed_labels
        # Synonym table unified to lowercase keys
        self.label_synonyms = {k.lower(): v for k, v in (label_synonyms or {}).items()}
        self.numeric_label_order = numeric_label_order
        self.use_json_mode = use_json_mode

        # Enhanced output constraint system prompt
        self.SYSTEM_PROMPT = (
            "You are a spectral chemometrics and pattern recognition expert. "
            "Task: given few-shot examples (x and label), classify the test samples. "
            "Output MUST be a JSON array of labels with length exactly equal to the number of test samples. "
            "Do NOT include any extra text or code fences."
        )

    def format_few_shot(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        return "\n".join(f"{e.get('name','ex')} : x={e['x']}, label={e['label']}" for e in few)

    def format_eval_prompt(self, eval_examples):
        xs = [e['x'] for e in eval_examples]
        return (
            "Known examples are above. "
            "Now classify the following samples. "
            "Return ONLY a JSON array, e.g.: [\"A\",\"B\",...]. "
            f"Samples: {xs}"
        )

    def chat_predict(self, messages):
        kwargs = dict(model=self.model, messages=messages, temperature=self.TEMPERATURE)
        # If backend supports OpenAI's JSON mode, can enable
        if self.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}  # Some backends need to return object, then extract specific field
        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        text = text.strip()

        # Use robust parser
        preds = robust_parse_predictions(
            text=text,
            n_expected=len(self.test_data),
            allowed_labels=self.allowed_labels,
            label_synonyms=self.label_synonyms,
            numeric_to_label=self.numeric_label_order,
        )

        # Token usage (different SDK field names may vary)
        try:
            prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
            print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
        except Exception:
            print("Token usage information not available")

        return preds, text

    def evaluate(self, preds: List[str], truths: List[str]) -> Tuple[float, List[int]]:
        n = min(len(preds), len(truths))
        wrong = [i for i in range(n) if preds[i] != truths[i]]
        if len(preds) != len(truths):
            print(f"[WARN] prediction length {len(preds)} != ground-truth length {len(truths)}; "
                  f"evaluating on first {n} samples.")
        acc = 1.0 - len(wrong) / n if n > 0 else 0.0
        return acc, wrong

    def run(self):
        print("=== Initial test set evaluation ===")
        initial_messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Few-shot examples:\n{self.format_few_shot(self.train_data)}\n\n"
                f"{self.format_eval_prompt(self.test_data)}"
            }
        ]
        print(".........",initial_messages)
        preds_test, raw_text = self.chat_predict(initial_messages)
        print('raw_model_output:', raw_text)
        print('parsed_preds:', preds_test)
        print('true_labels_test:', self.true_labels_test)
        test_acc, wrong_idx = self.evaluate(preds_test, self.true_labels_test)
        print('wrong indices:', wrong_idx)
        print(f"Test accuracy: {test_acc:.2%}")

from sklearn.metrics import r2_score

from sklearn.metrics import root_mean_squared_error
import re
class SpectrumReg:
    def __init__(
        self,
        dataset,
        api_key,
        base_url,
        model,
        temperature: float = 0.5,
        few_shot_initial: int = None,
    ):
        # Dataset splits from REG_Dataset
        self.train_data     = dataset.train_data   # list of {'name','x','y','split'}
        self.test_data      = dataset.test_data
        self.true_y_test    = dataset.y_test_true

        # LLM client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model

        # Hyperparameters
        self.TEMPERATURE             = temperature
        self.FEW_SHOT_INITIAL       = few_shot_initial

        # Prompts
        self.SYSTEM_PROMPT = (
            "You are a spectral chemometrics and regression expert. "
            "Based on provided examples of x→y, infer the relationship and predict y for new x. "
            "Use internal chain‐of‐thought but output only a Python list of predicted y values."
        )
        self.PROMPT_READ = (
            "Read the following structured data, each entry has 'name', 'x', 'y':\n{data}\n"
        )
        self.PROMPT_PREDICT = (
            "Predict 'y' for these new samples given only their 'x':\n{xs}\n"
            "Return only the list of predicted y values."
        )

    def format_read(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        return self.PROMPT_READ.format(data=few)

    def format_predict(self, examples):
        xs = [e['x'] for e in examples]
        return self.PROMPT_PREDICT.format(xs=xs)


    def chat(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.TEMPERATURE
        )
        raw = resp.choices[0].message.content.strip()

        # Print token usage
        if hasattr(resp, 'usage'):
            prompt_tokens = resp.usage.prompt_tokens if resp.usage.prompt_tokens else 0
            completion_tokens = resp.usage.completion_tokens if resp.usage.completion_tokens else 0
            print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
        else:
            print("Token usage information not available")

        # Remove ```python``` markers
        clean = raw.replace("```python\n", "").replace("\n```", "")

        # Split into token strings
        str_tokens = [tok.strip() for tok in clean.strip("[]").split(",")]

        # —— Added: convert each token to float —— 
        # If your tokens have extra single/double quotes, can strip first:
        float_preds = [
            float(tok.strip("'\""))  # Strip possible quotes then convert to float
            for tok in str_tokens
            if re.match(r"^-?\d+(\.\d+)?$", tok.strip("'\""))  # Only process tokens that look like numbers
        ]

        return float_preds, clean

    def evaluate(self, preds, truths):
        #print('preds:',preds)
        
        r2 = r2_score(truths, preds)
        # identify hardest by absolute error
        errors = [abs(p - t) for p, t in zip(preds, truths)]
        hard = sorted(range(len(errors)), key=lambda i: errors[i], reverse=True)
        return r2, hard


    def run(self):
        # 1) Baseline test evaluation
        print("=== Initial Test Evaluation ===")
        msgs = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        msgs.append({"role": "user", "content": self.format_read(self.train_data)})
        msgs.append({"role": "user", "content": self.format_predict(self.test_data)})
        preds, _ = self.chat(msgs)
        test_r2, _ = self.evaluate(preds, self.true_y_test)
        print(f"Test R²: {test_r2:.4f}")
        # RMSE for initial test
        init_rmse = root_mean_squared_error(self.true_y_test, preds)
        print(f"Test RMSE: {init_rmse:.4f}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectrum_anomaly_agent.py

An LLM‐driven multi‐round anomaly detection pipeline using few‐shot reasoning.
Train on normal examples only, validate and test on mixed normal/anomaly sets,
iteratively augment hard examples and early‐stop based on ROC‐AUC.
"""

import copy
from openai import OpenAI
from sklearn.metrics import precision_score, roc_auc_score

class SpectrumAno:
    def __init__(
        self,
        dataset,             # instance of ANO_Dataset
        api_key: str,
        base_url,
        model,
        temperature: float = 0.5,
        few_shot_initial: int = None,
    ):
        # Data splits
        self.train_data    = dataset.train_data + dataset.val_data
        self.test_data     = dataset.test_data
        self.y_test_true   = dataset.y_test
        # LLM client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model

        # Hyperparameters
        self.TEMPERATURE              = temperature
        self.FEW_SHOT_INITIAL        = few_shot_initial

        self.SYSTEM_PROMPT = (
    "You are a spectroscopic anomaly‐detection expert. "
    "Given a few labeled examples (True for anomaly, False for normal), "
    "perform chain‑of‑thought reasoning but output only a Python list of True/False."
)

        self.READ_PROMPT = (
    "Here are {n_train} labeled examples (name, x, label):\n{data}\n"
    "Note: True indicates anomaly; False indicates normal."
)

        self.PREDICT_PROMPT = (
    "You have {n_pred} new samples (only the feature vectors x are provided) in the same format. "
    "There are exactly {n_pred} samples, and you must return exactly {n_pred} Boolean values.  "
    "Respond with a single Python list of length {n_pred} containing only True or False, "
    "in the same order as the input samples, with no additional text or punctuation.\n"
    "Sample feature vectors:\n{xs}"
)


    def format_read(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        # Format as (name, x, label)
        lines = []
        for e in few:
            lbl = True if str(e['label']).lower() == 'true' else False
            lines.append(f"Name: {e['name']}, x: {e['x']}, label: {lbl}")
        return self.READ_PROMPT.format(n_train=len(few), data="\n".join(lines))

    def format_predict(self, examples):
        xs = [e['x'] for e in examples]
        return self.PREDICT_PROMPT.format(
            n_pred=len(examples),
            xs="\n".join([f"{i}: {x}" for i, x in enumerate(xs)])
        )

    def evaluate(self, preds, truths):
        # Convert ground truth to boolean
        truths_bool = [True if str(t).lower() == 'true' else False for t in truths]
        precision = precision_score(truths_bool, preds)
        auc = roc_auc_score(truths_bool, preds)
        hard = [i for i, (p, t) in enumerate(zip(preds, truths_bool)) if p != t]
        return precision, auc, hard



    def chat_predict(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.TEMPERATURE
        )
        text = resp.choices[0].message.content.strip()
        preds = [tok.strip().lower() == 'true' for tok in text.strip("[]").split(",")]
        
        # Print token usage
        if hasattr(resp, 'usage'):
            prompt_tokens = resp.usage.prompt_tokens if resp.usage.prompt_tokens else 0
            completion_tokens = resp.usage.completion_tokens if resp.usage.completion_tokens else 0
            print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
        else:
            print("Token usage information not available")
        
        return preds, text

    def evaluate(self, preds, truths):
        """
        Compute precision and ROC‐AUC, identify hard examples by mismatch.
        Returns:
            precision (float), auc (float), hard_indices (list)
        """
        # print("preds:",len(preds))
        # print('content:',preds)
        precision = precision_score(truths, preds)
        # For binary predictions, use preds as scores for AUC
        auc = roc_auc_score(truths, preds)
        hard = [i for i, (p, t) in enumerate(zip(preds, truths)) if p != t]
        return precision, auc, hard

    def run(self):
        # 1) Initial test evaluation
        print("=== Initial Test Evaluation ===")
        msgs = [{"role":"system","content":self.SYSTEM_PROMPT}]
        msgs.append({"role":"user","content":self.format_read(self.train_data)})
        msgs.append({"role":"user","content":self.format_predict(self.test_data)})

        preds_test, _ = self.chat_predict(msgs)
        print('preds_test:', preds_test)
        print('true_y_test:', self.y_test_true)
        prec_test, auc_test, _ = self.evaluate(preds_test, self.y_test_true)
        print(f"Test Precision: {prec_test:.4f}, Test AUC: {auc_test:.4f}")