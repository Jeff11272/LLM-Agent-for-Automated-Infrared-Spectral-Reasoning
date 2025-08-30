import os
import copy
import json
import re
import numpy as np
from openai import OpenAI
from sklearn.metrics import r2_score, root_mean_squared_error, precision_score, roc_auc_score



from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional

def _ensure_list(x):
    return [] if x is None else x

def stratified_train_val_split(
    data: List[Dict[str, Any]],
    label_key: str = "label",
    val_ratio: float = 0.25,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按类别均衡（每类均匀抽样）从 data 中抽取约 val_ratio 比例作为验证集；其余为训练集。
    要求 data[i][label_key] 可用；若找不到则尝试使用 'y' 作为标签（适用于二分类/异常检测存布尔或0/1）。
    """
    rng = np.random.default_rng(seed)
    by_lab = defaultdict(list)
    for idx, e in enumerate(data):
        lab = e.get(label_key, e.get("y", None))
        by_lab[lab].append((idx, e))

    val_indices = set()
    for lab, items in by_lab.items():
        n = len(items)
        if n == 0:
            continue
        n_val = max(1, int(round(n * val_ratio)))  # 每类至少抽 1 个
        perm = rng.permutation(n)[:n_val]
        for j in perm:
            val_indices.add(items[j][0])

    train_split, val_split = [], []
    for i, e in enumerate(data):
        (val_split if i in val_indices else train_split).append(e)

    return train_split, val_split


def random_train_val_split(
    data: List[Dict[str, Any]],
    val_ratio: float = 0.25,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    随机切分（适用于回归没有“类别”概念的情况），总体 3:1（train:val）。
    """
    n = len(data)
    if n == 0:
        return [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    val_set = set(idx[:n_val].tolist())
    train_split = [data[i] for i in range(n) if i not in val_set]
    val_split   = [data[i] for i in range(n) if i in val_set]
    return train_split, val_split


# -*- coding: utf-8 -*-
import json
import re
import ast
import copy
import os
import random
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

# ---------------------------
# Helpers: list & split
# ---------------------------
def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def stratified_train_val_split(
    data: List[dict], label_key: str = "label", val_ratio: float = 0.25, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    轻量分层划分：按标签分组后，按比例抽样到验证集，其余到训练集。
    """
    rng = random.Random(seed)
    buckets = {}
    for e in data:
        buckets.setdefault(e.get(label_key, "__NONE__"), []).append(e)

    train, val = [], []
    for _, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        k = max(1, int(round(n * val_ratio))) if n > 1 else 0  # 确保每类尽量有验证样本
        v_part = items[:k]
        t_part = items[k:]
        for x in t_part:
            x2 = dict(x)
            x2["split"] = "train"
            train.append(x2)
        for x in v_part:
            x2 = dict(x)
            x2["split"] = "val"
            val.append(x2)
    return train, val

# ---------------------------
# Robust label parsing
# ---------------------------
def _normalize_label(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().strip("'\"“”‘’`").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _best_label_match(raw: str, allowed: Optional[List[str]], synonyms: Optional[Dict[str, str]]) -> str:
    x = _normalize_label(raw).lower()
    if synonyms:
        if x in synonyms:
            x = synonyms[x].lower()
    if not allowed:
        return x
    lower_map = {a.lower(): a for a in allowed}
    if x in lower_map:
        return lower_map[x]
    x_nospace = x.replace(" ", "")
    for a in allowed:
        if a.lower().replace(" ", "") == x_nospace:
            return a
    return lower_map.get(x, raw)

def _extract_code_blocks(text: str) -> List[str]:
    return [b.strip() for b in re.findall(r"```(?:json|python|txt|yaml|md)?\s*(.*?)```",
                                          text, flags=re.DOTALL | re.IGNORECASE) if b.strip()]

def _try_parse_list_like(s: str):
    # JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("labels", "prediction", "predictions", "y", "output", "result"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
    except Exception:
        pass
    # JSON 数组片段
    for m in re.finditer(r"\[[^\[\]]*?\]", s, flags=re.DOTALL):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if isinstance(obj, list):
                return obj
        except Exception:
            continue
    # Python 字面量
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
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
        m = re.match(r"^\s*(?:[-*]\s+|(\d+)[\.\)\-:]\s*)(.+)$", t)
        if m:
            label = m.group(2) if m.group(1) else t.lstrip("-* ").strip()
            lines.append(label); continue
        m2 = re.match(r"^\s*([^:]+)\s*:\s*(.+)$", t)
        if m2:
            lines.append(m2.group(2)); continue
        lines.append(t)
    return lines

def robust_parse_predictions(
    text: str,
    n_expected: Optional[int] = None,
    allowed_labels: Optional[List[str]] = None,
    label_synonyms: Optional[Dict[str, str]] = None,
    numeric_to_label: Optional[List[str]] = None,
) -> List[str]:
    if not text or not text.strip():
        return []
    # 1) 代码块
    blocks = _extract_code_blocks(text)
    raw_list = None
    for b in blocks:
        arr = _try_parse_list_like(b)
        if isinstance(arr, list) and arr:
            raw_list = arr; break
    # 2) 全文
    if raw_list is None:
        arr = _try_parse_list_like(text)
        if isinstance(arr, list) and arr:
            raw_list = arr
    # 3) 退化分割
    if raw_list is None:
        if "," in text and ("\n" not in text or text.index(",") < text.index("\n")):
            parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
        else:
            parts = _parse_lines(text)
        raw_list = parts

    cleaned = [_normalize_label(str(x)) for x in raw_list]

    # 数字索引→标签
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

    final = [_best_label_match(c, allowed_labels, label_synonyms) for c in cleaned]

    # 长度对齐
    if n_expected is not None:
        if len(final) > n_expected:
            final = final[:n_expected]
        elif len(final) < n_expected:
            final = final + ["UNKNOWN"] * (n_expected - len(final))
    return final

# ---------------------------
# Multi-round SpectrumCLS
# ---------------------------
class SpectrumCLS:
    def __init__(
        self,
        train_data: list,
        test_data: list = None,
        true_labels_test: Optional[list] = None,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        max_rounds: int = 5,
        temperature: float = 0.5,
        few_shot_initial: int = None,
        early_stopping_patience: int = 3,
        save: bool = False,
        split_seed: int = 42,
        val_ratio: float = 0.25,
        # 新增：鲁棒解析相关
        allowed_labels: Optional[List[str]] = None,
        label_synonyms: Optional[Dict[str, str]] = None,  # 小写键: 规范值
        numeric_label_order: Optional[List[str]] = None,  # e.g. ["A","B","C"] for 0/1/2
        use_json_mode: bool = False,  # 后端若支持可开启
    ):
        train_data = _ensure_list(train_data)
        split_train, split_val = stratified_train_val_split(
            train_data, label_key="label", val_ratio=val_ratio, seed=split_seed
        )
        self.train_data = split_train
        self.val_data   = split_val
        self.test_data  = _ensure_list(test_data)

        self.true_labels_val = [e.get("label") for e in self.val_data]
        if true_labels_test is not None:
            self.true_labels_test = true_labels_test
        else:
            self.true_labels_test = [e.get("label") for e in self.test_data]

        # 允许标签集合：未提供则从 train/val/test 推断
        if allowed_labels is None:
            labels_set = set()
            for e in (self.train_data + self.val_data + self.test_data):
                if isinstance(e, dict) and "label" in e and e["label"] is not None:
                    labels_set.add(str(e["label"]))
            if self.true_labels_test:
                for t in self.true_labels_test:
                    labels_set.add(str(t))
            allowed_labels = sorted(labels_set) if labels_set else None
        self.allowed_labels = allowed_labels
        self.label_synonyms = {k.lower(): v for k, v in (label_synonyms or {}).items()}
        self.numeric_label_order = numeric_label_order
        self.use_json_mode = use_json_mode

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.MAX_ROUNDS = max_rounds
        self.TEMPERATURE = temperature
        self.FEW_SHOT_INITIAL = few_shot_initial
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience
        self.save = save

        self.SYSTEM_PROMPT = (
            "You are a spectral chemometrics and pattern recognition expert. "
            "Given few-shot examples (feature vector x and label), classify the provided samples. "
            "Your OUTPUT MUST be a JSON array of labels whose length equals the number of samples. "
            "Do NOT include extra text or code fences."
        )
        self._init_prompt_logged = False

    # ---------- Prompt builders ----------
    def format_few_shot(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        return "\n".join(f"{e.get('name','ex')} : x={e['x']}, label={e['label']}" for e in few)

    def format_eval_prompt(self, eval_examples):
        xs = [e['x'] for e in eval_examples]
        return (
            "Use the known examples above to classify the following samples.\n"
            "Return ONLY a JSON array like [\"A\",\"B\",...].\n"
            f"Samples: {xs}"
        )

    # ---------- Model call with robust parsing ----------
    def chat_predict(self, messages):
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.TEMPERATURE,
            top_p=1,
        )
        if self.use_json_mode:
            # 某些服务支持 JSON 强约束；若不支持也不会影响兜底解析
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)

        # token usage
        prompt_tokens = 0
        completion_tokens = 0
        try:
            prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
            print(f"[CLS] Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
        except Exception:
            print("[CLS] Token usage information not available")

        text = (resp.choices[0].message.content or "").strip()

        preds = robust_parse_predictions(
            text=text,
            n_expected=len(messages[-1]["content"]) if False else None,  # 占位：不使用该路径
            allowed_labels=self.allowed_labels,
            label_synonyms=self.label_synonyms,
            numeric_to_label=self.numeric_label_order,
        )

        return preds, text, prompt_tokens, completion_tokens

    # ---------- Eval ----------
    def evaluate(self, preds: List[str], truths: List[str]) -> Tuple[float, List[int]]:
        n = min(len(preds), len(truths))
        wrong = [i for i in range(n) if preds[i] != truths[i]]
        if len(preds) != len(truths):
            print(f"[WARN] prediction length {len(preds)} != ground-truth length {len(truths)}; "
                  f"evaluating on first {n} samples.")
        acc = 1.0 - len(wrong) / n if n > 0 else 0.0
        print('wrong indices:', wrong)
        return acc, wrong

    # ---------- Hard example mining ----------
    def augment_with_hard_examples(self, wrong_indices):
        existing = {e.get("name") for e in self.train_data}
        for idx in wrong_indices:
            if idx >= len(self.val_data):
                continue
            e = self.val_data[idx]
            name_new = f"{e.get('name','sample')}_hard"
            if name_new in existing:
                continue
            self.train_data.append({
                "name": name_new,
                "x": e['x'],
                "label": e['label'],
                "split": "train"
            })
            existing.add(name_new)

    # ---------- Replay logging ----------
    def _maybe_log_init_prompt(self, history_for_replay, user_content):
        if not self._init_prompt_logged:
            history_for_replay.append({"role": "user", "content": user_content})
            self._init_prompt_logged = True

    # ---------- Main loop ----------
    def run(self):
        history_for_replay = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        best_train = copy.deepcopy(self.train_data)
        best_val_acc = -1.0
        no_improve = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for round_idx in range(1, self.MAX_ROUNDS + 1):
            print(f"\n=== Round {round_idx} validation ===")
            user_content = (
                f"Known examples:\n{self.format_few_shot(self.train_data)}\n\n"
                f"{self.format_eval_prompt(self.val_data)}"
            )

            v_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            preds_val, val_text, prompt_tokens, completion_tokens = self.chat_predict(v_messages)

            # 若解析列表长度与验证样本数不一致，则在此对齐
            preds_val = (preds_val or [])
            preds_val = robust_parse_predictions(
                text=json.dumps(preds_val, ensure_ascii=False),
                n_expected=len(self.val_data),
                allowed_labels=self.allowed_labels,
                label_synonyms=self.label_synonyms,
                numeric_to_label=self.numeric_label_order,
            )

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            self._maybe_log_init_prompt(history_for_replay, user_content)
            history_for_replay.append({
                "role": "assistant",
                "content": json.dumps({
                    "type": "model_output",
                    "round": round_idx,
                    "val_raw_text": val_text
                }, ensure_ascii=False)
            })

            print('preds_val:', preds_val)
            val_acc, wrong_idxs = self.evaluate(preds_val, self.true_labels_val)
            print(f"Validation accuracy: {val_acc:.2%}, wrong indices: {wrong_idxs}")

            if val_acc == 1.0:
                print("Validation perfect, stopping early.")
                best_val_acc = val_acc
                best_train = copy.deepcopy(self.train_data)
                break

            # 难例增强
            self.augment_with_hard_examples(wrong_idxs)

            # 反馈（人读 + JSON）
            def _format_hard_example_feedback(wrong_idxs, preds_val, max_items=50):
                human_items, json_items = [], []
                n = len(wrong_idxs)
                take = min(n, max_items)
                for k in range(take):
                    i = wrong_idxs[k]
                    if i >= len(self.val_data): 
                        continue
                    sample = self.val_data[i]
                    sample_name = sample.get('name', f'idx_{i}')
                    predict = preds_val[i] if i < len(preds_val) else None
                    true_label = (
                        self.true_labels_val[i]
                        if i < len(self.true_labels_val)
                        else sample.get('label')
                    )
                    human_items.append(
                        f"sample_name:{sample_name}, predict:{predict}, true_label:{true_label}"
                    )
                    json_items.append({
                        "sample_name": sample_name,
                        "predict": predict,
                        "true_label": true_label,
                    })
                more_suffix = f" | ... and {n - take} more" if n > take else ""
                human_readable = " | ".join(human_items) + more_suffix if human_items else ""
                json_payload = {"type": "feedback", "round": round_idx,
                                "hard_examples": json_items, "truncated": n > take, "total": n}
                return human_readable, json_payload

            human_feedback, json_feedback = _format_hard_example_feedback(wrong_idxs, preds_val, max_items=50)
            history_for_replay.append({
                "role": "user",
                "content": json.dumps(json_feedback, ensure_ascii=False)
            })

            # 早停/回滚
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train = copy.deepcopy(self.train_data)
                no_improve = 0
            else:
                no_improve += 1
                print(f"No improvement on validation ({no_improve}/{self.EARLY_STOPPING_PATIENCE}).")
                self.train_data = copy.deepcopy(best_train)
                if no_improve > self.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered based on validation.")
                    break

        # ---------- Final test ----------
        print("\n=== Final single-shot test evaluation using replayed train/val history ===")
        final_prompt = copy.deepcopy(history_for_replay)
        final_prompt.append({"role": "user", "content":
            f"Known examples:\n{self.format_few_shot(best_train)}\n\n{self.format_eval_prompt(self.test_data)}"
        })

        preds_final, final_text, p_f, c_f = self.chat_predict(final_prompt)
        preds_final = robust_parse_predictions(
            text=json.dumps(preds_final, ensure_ascii=False),
            n_expected=len(self.test_data),
            allowed_labels=self.allowed_labels,
            label_synonyms=self.label_synonyms,
            numeric_to_label=self.numeric_label_order,
        )

        print('preds_final:', preds_final)
        print('true_labels_test:', self.true_labels_test)
        total_prompt_tokens = total_prompt_tokens + p_f
        total_completion_tokens = total_completion_tokens + c_f

        final_acc, wrong_final = self.evaluate(preds_final, self.true_labels_test)
        print(f"Final best test accuracy (replay + single-shot): {final_acc:.2%}")
        print('total_prompt_tokens:', total_prompt_tokens)
        print('total_completion_tokens:', total_completion_tokens)

        if self.save:
            os.makedirs('history_prompt', exist_ok=True)
            with open('history_prompt/cls_final_prompt.json', 'w', encoding='utf-8') as f:
                json.dump(final_prompt, f, ensure_ascii=False, indent=2)
            print("Final prompt saved to history_prompt/cls_final_prompt.json")

    def verify(
        self,
        prompt_json_path: str = "history_prompt/cls_final_prompt.json",
        truths: Optional[List[Any]] = None,
        temperature: Optional[float] = 0.0
    ):
        if not os.path.isfile(prompt_json_path):
            raise FileNotFoundError(f"Prompt JSON not found: {prompt_json_path}")
        with open(prompt_json_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("Invalid prompt JSON: expected a list of message dicts.")

        old_temp = self.TEMPERATURE
        if temperature is not None:
            self.TEMPERATURE = temperature

        preds, raw_text, prompt_tokens, completion_tokens = self.chat_predict(messages)
        y_true = self.true_labels_test if truths is None else truths
        acc, wrong = self.evaluate(preds, y_true)
        print(f"[VERIFY-CLS] accuracy: {acc:.2%}, wrong indices: {wrong}")
        print(f"[VERIFY-CLS] Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}")

        self.TEMPERATURE = old_temp
        return {
            "preds": preds,
            "acc": acc,
            "wrong_indices": wrong,
            "raw": raw_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }


class SpectrumReg:
    def __init__(
        self,
        dataset=None,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_rounds: int = 5,
        temperature: float = 0.5,
        few_shot_initial: Optional[int] = None,
        early_stopping_patience: int = 1,
        hard_feedback_n: int = 3,
        save: bool = False,
        train_data: Optional[List[dict]] = None,
        val_data: Optional[List[dict]] = None,
        test_data: Optional[List[dict]] = None,
        y_val_true: Optional[List[float]] = None,
        y_test_true: Optional[List[float]] = None,
        split_seed: int = 42,
        val_ratio: float = 0.25
    ):
        # —— 数据装载：优先 dataset；否则使用直传
        if dataset is not None:
            _train = _ensure_list(getattr(dataset, "train_data", None))
            _test  = _ensure_list(getattr(dataset, "test_data", None))
            _ytest = getattr(dataset, "y_test_true", None)
        else:
            _train = _ensure_list(train_data)
            _test  = _ensure_list(test_data)
            _ytest = y_test_true
        
        # —— 若 val 为空：随机 3:1 切分（回归无类别）
        split_train, split_val = random_train_val_split(_train, val_ratio=val_ratio, seed=split_seed)
        self.train_data = split_train
        self.val_data   = split_val

        self.test_data = _test

        self.true_y_val = [e["y"] for e in self.val_data]

        if _ytest is not None:
            self.true_y_test = _ytest
        else:
            self.true_y_test = [e["y"] for e in self.test_data]

        # —— 其余保持不变
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.MAX_ROUNDS = max_rounds
        self.TEMPERATURE = temperature
        self.FEW_SHOT_INITIAL = few_shot_initial
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience
        self.HARD_FEEDBACK_N = hard_feedback_n
        self.save = save

        self.SYSTEM_PROMPT = (
            "You are a spectral chemometrics and regression expert. "
            "Given several examples of x → y, infer the mapping and predict numeric y for new x values. "
            "You may use internal chain-of-thought, but output only a Python list (e.g. [0.1, 1.2, -3.4]) of predicted y values."
        )
        self.PROMPT_READ = "Read these labeled examples (name, x, y):{data}"
        self.PROMPT_PREDICT = (
            "Predict 'y' for these samples given their 'name' and 'x'. Each entry is an object with 'name' first:{xs}"
            "Return only the Python list of numeric y values in the SAME ORDER as the provided list."
        )
        self._init_prompt_logged = False


    # --- Prompt formatting helpers ---
    def _json_dump_examples(self, examples, include_y=True):
        out = []
        for e in examples:
            item = {"name": e.get("name", ""), "x": e.get("x")}
            if include_y and "y" in e:
                item["y"] = e["y"]
            out.append(item)
        json_str = json.dumps(out, ensure_ascii=False, separators=(',', ':'))
        return json_str

    def format_read_train(self, examples):
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        json_data = self._json_dump_examples(few, include_y=True)
        clean_data = json_data.replace('\"', '"').replace('\\n', '')
        return self.PROMPT_READ.format(data=clean_data)

    def format_predict_xs(self, examples):
        simple_list = [{"name": e.get("name", ""), "x": e.get("x")} for e in examples]
        json_data = json.dumps(simple_list, ensure_ascii=False, separators=(',', ':'))
        clean_data = json_data.replace('\"', '"').replace('\\n', '')
        return self.PROMPT_PREDICT.format(xs=clean_data)

    # --- 仅首轮记录输入（train+val）的 prompt ---
    def _maybe_log_init_prompt(self, history_for_replay, user_content: str):
        if not self._init_prompt_logged:
            history_for_replay.append({"role": "user", "content": user_content})
            self._init_prompt_logged = True

    # --- Chat wrapper (returns parsed floats, raw text, token usage) ---
    def chat(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.TEMPERATURE,
            
            top_p=1
        )
        raw = resp.choices[0].message.content.strip()

        # Token usage
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(resp, "usage"):
            prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
            print(f"[CHAT] Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
        else:
            print("[CHAT] Token usage info not available")

        # Clean code fences if present
        clean = raw.replace("```python\n", "").replace("\n```", "").strip()

        # Extract numeric tokens robustly (support scientific notation)
        num_strs = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", clean)
        float_preds = [float(s) for s in num_strs]

        return float_preds, clean, prompt_tokens, completion_tokens

    # --- Evaluation ---
    def evaluate(self, preds: List[float], truths: List[float]):
        if len(preds) != len(truths):
            min_len = min(len(preds), len(truths))
            print(f"[WARN] Prediction length ({len(preds)}) != truth length ({len(truths)}). Truncating to {min_len}.")
            preds = preds[:min_len]
            truths = truths[:min_len]

        r2 = r2_score(truths, preds)
        errors = [abs(p - t) for p, t in zip(preds, truths)]
        hard_indices = sorted(range(len(errors)), key=lambda i: errors[i], reverse=True)
        return r2, hard_indices

    # --- Augmentation ---
    def augment_with_hard_examples(self, wrong_indices: List[int]):
        for idx in wrong_indices:
            e = self.val_data[idx]
            self.train_data.append({
                "name": f"{e.get('name','')}_hard",
                "x": e["x"],
                "y": e["y"],
                "split": "train"
            })

    # --- Main multi-round loop ---
    def run(self):
        print("=== Multi-round regression enhancement ===")
        # 仅用于回放：system +（首轮的）输入prompt + 每轮模型输出 + 每轮反馈
        history_for_replay = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        best_train = copy.deepcopy(self.train_data)
        best_val_r2 = -1.0
        no_improve = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for round_idx in range(1, self.MAX_ROUNDS + 1):
            print(f"\n--- Round {round_idx} validation ---")
            # 构造当轮推理用的消息（不会全量写入 history，只在首轮记录一次）
            train_read = self.format_read_train(self.train_data)
            val_predict = self.format_predict_xs(self.val_data)
            user_content = f"Known examples:{train_read}, {val_predict}"

            v_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            preds_val, raw_val_text, p_tokens, c_tokens = self.chat(v_messages)
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens

            # 仅首轮：记录一次输入 prompt（train+val）
            self._maybe_log_init_prompt(history_for_replay, user_content)

            # 每一轮：记录“模型输出”（assistant）
            model_output_payload = {
                "type": "model_output",
                "round": round_idx,
                "val_raw_text": raw_val_text,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens
            }
            history_for_replay.append({
                "role": "assistant",
                "content": json.dumps(model_output_payload, ensure_ascii=False)
            })

            # 评估
            val_r2, hard_idxs = self.evaluate(preds_val, self.true_y_val)
            print(f"Validation R²: {val_r2:.4f}, top hard indices: {hard_idxs[:]}")

            # 每一轮：记录“反馈内容”（保持原有格式）
            n = self.HARD_FEEDBACK_N
            feedback_lines = []
            for i in hard_idxs[:n]:
                pred_val = preds_val[i] if i < len(preds_val) else None
                true_val = self.true_y_val[i]
                diff = abs(pred_val - true_val) if pred_val is not None else None
                feedback_lines.append(
                    f"{self.val_data[i].get('name','')}: prediction: {pred_val}, "
                    f"true value: {true_val}, difference: {diff}"
                )
            feedback_text = "Hard examples feedback (top {}):\n{}".format(
                n, "\n".join(feedback_lines) if feedback_lines else "None"
            )
            history_for_replay.append({"role": "user", "content": feedback_text})

            # （可选）难例增强
            self.augment_with_hard_examples(hard_idxs[:n])

            # 早停 / 回滚
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_train = copy.deepcopy(self.train_data)
                no_improve = 0
            else:
                no_improve += 1
                print(f"No improvement on validation ({no_improve}/{self.EARLY_STOPPING_PATIENCE}). Rolling back to best train set.")
                self.train_data = copy.deepcopy(best_train)
                if no_improve > self.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered based on validation.")
                    break

            if val_r2 == 1.0:
                print("Validation perfect (R^2=1.0), stopping early.")
                break

        # --- Final: 单次测试，用回放历史（首轮输入 + 每轮模型输出 + 每轮反馈） ---
        print("\n=== Final single-shot test evaluation using replayed train/val history ===")
        final_prompt = copy.deepcopy(history_for_replay)
        final_prompt.append({
            "role": "user",
            "content": (
                "Known examples:\n"
                f"{self.PROMPT_READ.format(data=self._json_dump_examples(best_train, include_y=True))}\n\n"
                f"{self.format_predict_xs(self.test_data)}"
            )
        })

        preds_final, final_raw, p_f, c_f = self.chat(final_prompt)
        print('preds_final:', preds_final)
        total_prompt_tokens += p_f
        total_completion_tokens += c_f

        final_r2, _ = self.evaluate(preds_final, self.true_y_test)
        final_rmse = root_mean_squared_error(self.true_y_test[:len(preds_final)], preds_final)
        print(f"Final Test R²: {final_r2:.4f}")
        print(f"Final Test RMSE: {final_rmse:.4f}")
        print(f"Total tokens used - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}")

        if self.save:
            os.makedirs("history_prompt", exist_ok=True)
            with open("history_prompt/reg_final_prompt.json", "w", encoding="utf-8") as f:
                json.dump(final_prompt, f, ensure_ascii=False, indent=2)
            print("Final prompt saved to history_prompt/reg_final_prompt.json")

        return {
            "final_preds": preds_final,
            "final_raw": final_raw,
            "final_r2": final_r2,
            "final_rmse": final_rmse,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens
        }

    # --- Verification utility (run saved prompt) ---
    def verify(self, prompt_json_path: str = "history_prompt/reg_final_prompt.json", truths: Optional[List[Any]] = None, temperature: Optional[float] = 0.0):
        if not os.path.isfile(prompt_json_path):
            raise FileNotFoundError(f"Prompt JSON not found: {prompt_json_path}")
        with open(prompt_json_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("Invalid prompt JSON: expected a list of message dicts.")

        old_temp = self.TEMPERATURE
        if temperature is not None:
            self.TEMPERATURE = temperature

        preds, raw_text, p_tokens, c_tokens = self.chat(messages)
        y_true = self.true_y_test if truths is None else truths
        r2, wrong = self.evaluate(preds, y_true)
        print(f"[VERIFY-REG] R²: {r2:.4f}, wrong indices: {wrong}")
        print(f"[VERIFY-REG] Tokens - Prompt: {p_tokens}, Completion: {c_tokens}")

        self.TEMPERATURE = old_temp
        return {
            "preds": preds,
            "r2": r2,
            "wrong_indices": wrong,
            "raw": raw_text,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens
        }


import os
import re
import json
import copy
from typing import List, Tuple, Optional, Any
from openai import OpenAI
from sklearn.metrics import precision_score, roc_auc_score

class SpectrumAno:
    def __init__(self, dataset, 
                 api_key: str,
                 base_url: str, 
                 model: str,
                 max_rounds: int = 5, 
                 temperature: float = 0.5,
                 few_shot_initial: int = None, 
                 early_stopping_patience: int = 2,
                 save: bool = False,
                 split_seed: int = 42,
                 val_ratio: float = 0.25 
                 ):
        # 原始数据
        _train = _ensure_list(getattr(dataset, "train_data", None))
        _test  = _ensure_list(getattr(dataset, "test_data", None))
        _ytest = getattr(dataset, "y_test", None)

        # —— 若 val 为空：按“类别均衡 + 3:1”自动切分
        split_train, split_val = stratified_train_val_split(
            _train, label_key="label", val_ratio=val_ratio, seed=split_seed
        )
        self.train_data = split_train
        self.val_data   = split_val

        self.test_data = _test

        self.y_val_true = [e.get("label", e.get("y", False)) for e in self.val_data]

        if _ytest is not None:
            self.y_test_true = _ytest
        else:
            self.y_test_true = [e.get("label", e.get("y", False)) for e in self.test_data]

        # 其余保持不变
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.MAX_ROUNDS = max_rounds
        self.TEMPERATURE = temperature
        self.FEW_SHOT_INITIAL = few_shot_initial
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience
        self.save = save

        self._init_prompt_logged = False

        self.SYSTEM_PROMPT = (
            "You are an expert in spectroscopic anomaly detection. "
            "Each sample is a numerical feature vector. "
            "IMPORTANT: Do NOT provide any explanation or reasoning. "
            "When asked to predict, you MUST RETURN EXACTLY a JSON array of boolean values "
            "using lowercase `true` and `false` (for example: [true, false, true]) with length equal to the number of query samples. "
            "Return nothing else."
        )
        self.READ_PROMPT = (
            "Here are {n_train} labeled examples (name, x, label):\n{data}\n"
            "Note: label true indicates anomaly; false indicates normal."
        )
        self.PREDICT_PROMPT = (
            "You have {n_pred} new samples (only feature vectors x are provided). "
            "Return EXACTLY a JSON array of length {n_pred} containing only lowercase true or false, in the same order. "
            "Return NOTHING else.\n"
            "Example (for 2 samples): [true, false]\n"
            "Sample feature vectors:\n{xs}"
        )


    # ---------------- formatting helpers ----------------
    def _label_to_text(self, lab: Any) -> str:
        if isinstance(lab, bool):
            return "true" if lab else "false"
        if isinstance(lab, (int, float)):
            return "true" if int(lab) == 1 else "false"
        if isinstance(lab, str):
            s = lab.strip().lower()
            return "true" if s in ("true", "1", "t", "yes") else "false"
        return "false"

    def format_read(self, examples: List[dict]) -> str:
        few = examples if self.FEW_SHOT_INITIAL is None else examples[:self.FEW_SHOT_INITIAL]
        lines = []
        for e in few:
            lab = e.get("label", e.get("y", None))
            lab_text = self._label_to_text(lab)
            lines.append(f"Sample '{e.get('name','?')}': features={e['x']}, label={lab_text}")
        return self.READ_PROMPT.format(n_train=len(few), data="\n".join(lines))

    def format_predict(self, examples: List[dict]) -> str:
        xs = [e['x'] for e in examples]
        return self.PREDICT_PROMPT.format(n_pred=len(examples), xs=xs)

    # ---------------- parsing & model call ----------------
    def _parse_bool_list_from_raw(self, raw: str, n_pred: int) -> Optional[List[bool]]:
        s = raw.replace("，", ",").replace("```", "").strip()

        # 1) direct json load
        try:
            j = json.loads(s)
            if isinstance(j, list):
                parsed = []
                for v in j:
                    if isinstance(v, bool):
                        parsed.append(v)
                    elif isinstance(v, (int, float)):
                        parsed.append(bool(v))
                    elif isinstance(v, str):
                        lv = v.strip().lower()
                        parsed.append(lv in ("true", "t", "1", "yes"))
                    else:
                        return None
                return parsed
        except Exception:
            pass

        # 2) extract first bracket content and try json
        m = re.search(r"\[[^\]]*\]", s, flags=re.S)
        if m:
            try:
                j = json.loads(m.group(0))
                if isinstance(j, list):
                    parsed = []
                    for v in j:
                        if isinstance(v, bool):
                            parsed.append(v)
                        elif isinstance(v, (int, float)):
                            parsed.append(bool(v))
                        elif isinstance(v, str):
                            lv = v.strip().lower()
                            parsed.append(lv in ("true", "t", "1", "yes"))
                        else:
                            return None
                    return parsed
            except Exception:
                pass

        # 3) extract true/false/1/0 tokens
        tokens = re.findall(r"\b(true|false|1|0)\b", s, flags=re.I)
        if tokens:
            parsed = []
            for tok in tokens:
                tt = tok.lower()
                parsed.append(True if tt in ("true", "1") else False)
            return parsed

        return None

    def _call_model_once(self, messages: List[dict], temperature: float) -> Tuple[str, int, int]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=16000,
            top_p=1
        )
        usage = getattr(resp, "usage", None)
        p_t = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        c_t = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        raw = resp.choices[0].message.content.strip()
        return raw, p_t, c_t

    def chat_predict(self, messages: List[dict], n_pred: Optional[int] = None, max_retries: int = 2) -> Tuple[List[bool], str, int, int]:
        attempt = 0
        acc_p = 0
        acc_c = 0
        cur_msgs = list(messages)

        while True:
            attempt += 1
            temp = self.TEMPERATURE if attempt == 1 else 0.0
            raw, p_t, c_t = self._call_model_once(cur_msgs, temperature=temp)
            acc_p += p_t
            acc_c += c_t

            parsed = None
            if n_pred is not None:
                parsed = self._parse_bool_list_from_raw(raw, n_pred)
                if parsed is not None and len(parsed) == n_pred:
                    return parsed, raw, acc_p, acc_c
            else:
                parsed = self._parse_bool_list_from_raw(raw, 0)
                if parsed is not None:
                    return parsed, raw, acc_p, acc_c

            if attempt <= max_retries:
                enforce = {
                    "role": "user",
                    "content": (
                        "Your previous reply did not follow the required format. "
                        "NOW RESPOND WITH EXACTLY a JSON array of length "
                        f"{n_pred} containing only lowercase true or false (e.g. [true, false]). "
                        "Return absolutely nothing else."
                    )
                }
                example = {"role": "user", "content": "Example (for 3 samples): [true, false, true]"}
                cur_msgs = cur_msgs + [enforce, example]
                continue

            if parsed is not None:
                if n_pred is not None:
                    if len(parsed) < n_pred:
                        parsed = parsed + [False] * (n_pred - len(parsed))
                    else:
                        parsed = parsed[:n_pred]
                return parsed, raw, acc_p, acc_c

            print("[ERROR] Unable to parse model output into boolean list. Returning all-False fallback. Raw output:")
            print(raw)
            fallback = [False] * (n_pred or 0)
            return fallback, raw, acc_p, acc_c

    # ---------------- evaluation / augmentation ----------------
    def _normalize_truths(self, truths: List[Any]) -> List[bool]:
        out = []
        for t in truths:
            if isinstance(t, bool):
                out.append(t)
            elif isinstance(t, (int, float)):
                out.append(bool(int(t)))
            elif isinstance(t, str):
                out.append(t.strip().lower() in ("1", "true", "t", "yes"))
            else:
                out.append(False)
        return out

    def evaluate(self, preds: List[bool], truths: List[Any]) -> Tuple[float, float, List[int]]:
        truths_bool = self._normalize_truths(truths)
        if len(preds) != len(truths_bool) or len(truths_bool) == 0:
            return 0.0, 0.0, list(range(min(len(preds), len(truths_bool))))
        print('preds:', preds)
        print('truths_bool:', truths_bool)
        precision = precision_score(truths_bool, preds)
        auc = roc_auc_score(truths_bool, preds)
        hard = [i for i, (p, t) in enumerate(zip(preds, truths_bool)) if p != t]
        return precision, auc, hard

    def augment_hard(self, wrong_idxs: List[int]):
        for idx in wrong_idxs:
            e = self.val_data[idx]
            e_hard = copy.deepcopy(e)
            self.train_data.append(e_hard)

    # ---------------- run (val-driven multi-round + final single-shot test) ----------------
    def run(self):
        total_prompt_tokens = 0
        total_completion_tokens = 0

        best_train = copy.deepcopy(self.train_data)
        best_auc = float("-inf")
        no_improve = 0

        # 仅用于回放：system +（首轮输入）+ 每轮模型输出 + 每轮反馈
        history_for_replay = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        for rd in range(1, self.MAX_ROUNDS + 1):
            print(f"\n=== Round {rd} Validation ===")
            user_read = self.format_read(self.train_data)
            user_pred_val = self.format_predict(self.val_data)

            # 每轮推理消息使用“新鲜的”system+当轮user，不叠加旧的大块prompt
            v_msgs = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_read},
                {"role": "user", "content": user_pred_val}
            ]

            preds_val, reply, p_t, c_t = self.chat_predict(v_msgs, n_pred=len(self.val_data), max_retries=2)
            total_prompt_tokens += p_t
            total_completion_tokens += c_t

            # ——仅首轮：记录一次输入 prompt（train+val）
            if not self._init_prompt_logged:
                history_for_replay.append({"role": "user", "content": user_read})
                history_for_replay.append({"role": "user", "content": user_pred_val})
                self._init_prompt_logged = True

            # ——每一轮：记录“模型输出”（assistant），包装为 JSON，便于回放与统计
            model_output_payload = {
                "type": "model_output",
                "round": rd,
                "val_raw_text": reply,
                "prompt_tokens": p_t,
                "completion_tokens": c_t
            }
            history_for_replay.append({
                "role": "assistant",
                "content": json.dumps(model_output_payload, ensure_ascii=False)
            })

            # 评估
            print('preds_val:', preds_val)
            prec_val, auc_val, hard_idxs = self.evaluate(preds_val, self.y_val_true)
            print(f"Validation Precision: {prec_val:.4f}, AUC: {auc_val:.4f}, Hard count: {len(hard_idxs)}")

            if auc_val >= 1.0:
                print("Perfect validation AUC; stopping early.")
                best_train = copy.deepcopy(self.train_data)
                # 仍需记录当轮反馈（即使 perfect）
                pass

            self.augment_hard(hard_idxs)

            # ——每一轮：记录“反馈内容”（格式保持不变）
            fb_entries = []
            for i in hard_idxs[:3]:
                sample_name = self.val_data[i]['name']
                pred_value = preds_val[i]
                true_value = self.y_val_true[i]
                fb_entries.append(f"Sample '{sample_name}': Predict: {pred_value}, True value: {true_value}")
            fb_msg = "Augmented hard examples:\n" + ("\n".join(fb_entries) if fb_entries else "None")
            history_for_replay.append({"role": "user", "content": fb_msg})

            # 早停/回滚（基于 AUC）
            if auc_val > best_auc:
                best_auc = auc_val
                best_train = copy.deepcopy(self.train_data)
                no_improve = 0
            else:
                no_improve += 1
                print(f"No improvement on validation ({no_improve}/{self.EARLY_STOPPING_PATIENCE}).")
                self.train_data = copy.deepcopy(best_train)
                if no_improve > self.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered based on validation.")
                    break

            if auc_val >= 1.0:
                break

        # Final single-shot test evaluation using replay-only history + best_train
        print("\n=== Final single-shot test evaluation using replay-only history ===")
        final_prompt = copy.deepcopy(history_for_replay)
        final_prompt.append({"role": "user", "content": self.format_read(best_train)})
        final_prompt.append({"role": "user", "content": self.format_predict(self.test_data)})

        old_temp = self.TEMPERATURE
        self.TEMPERATURE = 0.0
        preds_final, final_raw, p_f, c_f = self.chat_predict(final_prompt, n_pred=len(self.test_data), max_retries=2)
        self.TEMPERATURE = old_temp

        total_prompt_tokens += p_f
        total_completion_tokens += c_f

        print('preds_final:', preds_final)
        prec_final, auc_final, hard_final = self.evaluate(preds_final, self.y_test_true)
        print(f"\nFinal best Test Precision/AUC: {prec_final:.4f} / {best_auc:.4f}")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")

        # Optionally save replay-only prompt template (without test)
        if self.save:
            os.makedirs("history_prompt", exist_ok=True)
            save_msgs = copy.deepcopy(history_for_replay)
            save_msgs.append({"role": "user", "content": self.format_read(best_train)})
            path = "history_prompt/ano_final_prompt.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(save_msgs, f, ensure_ascii=False, indent=2)
            print(f"[ANO] Saved replay-only prompt template -> {path}")

    # ---------------- verify helper ----------------
    def verify(
        self,
        prompt_json_path: str = "history_prompt/ano_final_prompt.json",
        truths: Optional[List[bool]] = None,
        temperature: Optional[float] = 0.0,
        n_pred: Optional[int] = None
    ):
        if not os.path.isfile(prompt_json_path):
            raise FileNotFoundError(f"Prompt JSON not found: {prompt_json_path}")
        with open(prompt_json_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("Invalid prompt JSON: expected a list of message dicts.")

        if n_pred is None:
            n_pred = len(self.test_data)

        old_temp = self.TEMPERATURE
        if temperature is not None:
            self.TEMPERATURE = temperature

        preds, raw_text, p_t, c_t = self.chat_predict(
            messages + [{"role": "user", "content": self.format_predict(self.test_data)}],
            n_pred=n_pred, max_retries=2
        )

        y_true = self.y_test_true if truths is None else truths
        prec, auc, hard = self.evaluate(preds, y_true)
        print(f"[VERIFY-ANO] Precision: {prec:.4f}, AUC: {auc:.4f}, hard mismatches: {len(hard)}")
        print(f"[VERIFY-ANO] Tokens - Prompt: {p_t}, Completion: {c_t}")

        self.TEMPERATURE = old_temp
        return {
            "preds": preds,
            "precision": prec,
            "auc": auc,
            "hard_indices": hard,
            "raw": raw_text,
            "prompt_tokens": p_t,
            "completion_tokens": c_t
        }

