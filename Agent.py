import os
import json
import re
import numpy as np
from openai import OpenAI
# from  sktime.datasets import load_tecator
import Preprocess_method as Preprocess
import Feature_extract as FeatureExt
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union


ALL_PREPROCESS_FUNCS = [
    "baseline_correction_asls",
    "savitzky_golay_smoothing",
    "standard_normal_variate",
    "multiplicative_scatter_correction",
    "normalization_min_max",
    "detrend_spectrum",
    "snv_fd",
    "sg_fd",
    "msc_fd"
]
ALL_FEATURE_FUNCS = [
    "pca_feature_extraction",
    "nmf_feature_extraction",
    "cwt_feature_extraction",
    "spectral_derivative",
    "peak_feature_extraction",
    "statistical_feature_extraction",
    "lambert_pearson_feature_extraction",
    "Partial_Least_Squares"
]


def _canon_pipeline(pre: List[str], feat: List[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Treat (pre_list, feat_list) as an indivisible pipeline key, maintaining order.
    Used as a key for Counter.
    """
    return (tuple(pre), tuple(feat))


class SpectralAgent:
    def __init__(self, api_key, base_url, model_name, regression_label=None,):
        self.api_key = api_key
        self.base_url = base_url
        self.regression_label = regression_label
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.ALL_PREPROCESS_FUNCS = ALL_PREPROCESS_FUNCS
        self.ALL_FEATURE_FUNCS = ALL_FEATURE_FUNCS
        self.model_name = model_name

        self.SYSTEM_PROMPT = f"""
You are a spectral analysis decision agent.
You will receive a list of papers, each including "paper_name", "preprocessing_method" and "feature_extracting_method" descriptions.
For each paper, map its descriptions to one or more valid function names below.
Output a JSON object mapping each paper to its:
  "preprocessing": [...],
  "features": [...]
Use only names from the available lists; if a paper has no methods or all its methods are invalid, you may assign an empty list.
Example output:
{{
  "paper_A": {{"preprocessing": ["savitzky_golay_smoothing"], "features": ["pca_feature_extraction"]}},
  "paper_B": {{"preprocessing": [], "features": []}}
}}

Available preprocessing function names:
  {self.ALL_PREPROCESS_FUNCS}
Available feature-extraction function names:
  {self.ALL_FEATURE_FUNCS}

Rules:
 1. Map keywords in descriptions to exact function names.
 2. Do not invent names not in the lists.
 3. Output only valid JSON and nothing else.
"""

    def decide_methods_per_paper(self, papers_info: list) -> dict:
        messages = [
            {"role": "system",  "content": self.SYSTEM_PROMPT},
            {"role": "user",    "content": "papers_info:\n" + json.dumps(papers_info, ensure_ascii=False, indent=2)}
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        text = completion.choices[0].message.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(json)?", "", text)
            text = re.sub(r"```$", "", text)
            text = text.strip()
        start = text.find('{')
        if start == -1:
            raise RuntimeError(f"No JSON object found in response: {text}")
        depth = 0
        end = None
        for idx, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            if depth == 0:
                end = start + idx + 1
                break
        if end is None:
            raise RuntimeError(f"Incomplete JSON object in response: {text}")
        json_text = text[start:end]
        try:
            decision = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON: {e}\nExtracted text: {json_text}")
        return decision

    def process_all_papers(self, data: np.ndarray, methods_map: dict) -> dict:
        PRE_MAP = {fn: getattr(Preprocess, fn) for fn in self.ALL_PREPROCESS_FUNCS}
        FEAT_MAP = {fn: getattr(FeatureExt, fn) for fn in self.ALL_FEATURE_FUNCS}

        global_pre = []
        global_feat = []
        for cfg in methods_map.values():
            valid_pre = [fn for fn in cfg.get("preprocessing", []) if fn in PRE_MAP]
            valid_feat = [fn for fn in cfg.get("features", []) if fn in FEAT_MAP]
            global_pre.extend(valid_pre)
            global_feat.extend(valid_feat)
        global_pre = list(dict.fromkeys(global_pre))
        global_feat = list(dict.fromkeys(global_feat))

        fallback_pre = global_pre or []
        fallback_feat = global_feat or ["pca_feature_extraction"]

        result_map = {}
        for paper, cfg in methods_map.items():
            valid_pre = [fn for fn in cfg.get("preprocessing", []) if fn in PRE_MAP]
            valid_feat = [fn for fn in cfg.get("features", []) if fn in FEAT_MAP]
            pre_list = valid_pre or fallback_pre
            feat_list = valid_feat or fallback_feat

            processed = data.copy()
            for fn in pre_list:
                processed = PRE_MAP[fn](processed)

            feats_dict = {}
            for fn in feat_list:
                if fn == "lambert_pearson_feature_extraction":
                    if self.regression_label is None:
                        raise ValueError("lambert_label_path must be set for lambert_pearson_feature_extraction")
                    target = self.regression_label
                    feats_dict[fn] = FEAT_MAP[fn](processed, target)
                elif fn == "Partial_Least_Squares":
                    target = self.regression_label.reshape([-1])
                    processed = processed.reshape((processed.shape[1], processed.shape[2]))
                    feats_dict[fn] = FEAT_MAP[fn](processed, target)
                else:
                    feats_dict[fn] = FEAT_MAP[fn](processed)

            result_map[paper] = {
                "processed_data": processed,
                "extracted_features": feats_dict
            }

        return result_map

    # -------------------- Method selection related (overall voting version) --------------------

    def _is_valid_methods_for_paper(
        self,
        methods_map: Dict[str, Dict[str, List[str]]],
        paper: str,
        pre_funcs: set,
        feat_funcs: set,
        require_both: bool = True,
    ) -> bool:
        """
        Determine if a paper's methods are "complete and valid":
        - require_both=True: Both preprocessing and features must be non-empty and have valid function names
        - require_both=False: Either side being non-empty and valid is sufficient
        """
        mm = methods_map.get(paper, {})
        pre = [fn for fn in mm.get("preprocessing", []) if fn in pre_funcs]
        feat = [fn for fn in mm.get("features", []) if fn in feat_funcs]
        if require_both:
            return len(pre) > 0 and len(feat) > 0
        else:
            return len(pre) > 0 or len(feat) > 0

    def _first_valid_paper(
        self,
        methods_map: Dict[str, Dict[str, List[str]]],
        candidates: List[str],
        pre_funcs: set,
        feat_funcs: set,
        require_both: bool = True,
    ) -> Optional[str]:
        """Find the first "complete and valid" paper in candidates order; return None if not found."""
        for p in candidates:
            if self._is_valid_methods_for_paper(methods_map, p, pre_funcs, feat_funcs, require_both):
                return p
        return None

    def select_methods(
        self,
        methods_map: Dict[str, Dict[str, List[str]]],
        k: int = 3,
        paper_order: Optional[List[str]] = None,
        *,
        # Free choice: can pass paper name or index in paper_order (both 0-based and 1-based supported)
        prefer_paper: Optional[Union[str, int]] = None,
        prefer_strict: bool = False,   # True raises error if specified paper is invalid; False falls back to majority voting
        require_both: bool = True,     # True requires both preprocessing and features; False requires either side to be non-empty
        min_support: int = 1,          # Minimum support for majority vote; setting to 2 is more stable, 1 means no threshold
    ) -> Dict[str, List[str]]:
        """
        Method selector (overall pipeline voting):

        Rule priority:
        1) If prefer_paper specifies a paper (by name or index), prioritize using that paper's "paired methods" (preprocessing + features used together).
           - If prefer_strict=True and the paper's methods are incomplete/invalid, raise ValueError;
           - If prefer_strict=False and the paper's methods are incomplete/invalid, fall back to rules 2/3.
        2) Among the first k papers, perform majority voting using "(pre_list, feat_list)" pairs.
           - Only adopt when there is a unique winner with votes >= min_support;
           - If tied or no valid votes, proceed to 3).
        3) Fallback: Find the first "complete and valid" paper among the first k in order, and use its paired methods;
           - If still not found, return {"preprocessing": [], "features": ["pca_feature_extraction"]} as final fallback.
        """
        PRE_FUNCS = set(self.ALL_PREPROCESS_FUNCS)
        FEAT_FUNCS = set(self.ALL_FEATURE_FUNCS)

        # Paper order and top-k
        if paper_order is None:
            paper_order = list(methods_map.keys())
        topk_papers = [p for p in paper_order if p in methods_map][:max(1, k)]

        # ---------- Manual priority specification ----------
        def _pick_from(pname: str) -> Optional[Dict[str, List[str]]]:
            mm = methods_map.get(pname, {})
            pre = [x for x in mm.get("preprocessing", []) if x in PRE_FUNCS]
            feat = [x for x in mm.get("features", []) if x in FEAT_FUNCS]
            ok = (len(pre) > 0 and len(feat) > 0) if require_both else (len(pre) > 0 or len(feat) > 0)
            if ok:
                return {"preprocessing": pre, "features": feat or ["pca_feature_extraction"]}
            return None

        if prefer_paper is not None:
            if isinstance(prefer_paper, int):
                # Support 0/1-based indexing
                idx_candidates = []
                if 0 <= prefer_paper < len(paper_order):
                    idx_candidates.append(prefer_paper)
                if 1 <= prefer_paper <= len(paper_order):
                    idx_candidates.append(prefer_paper - 1)
                picked = None
                for idx in idx_candidates:
                    name_i = paper_order[idx]
                    cand = _pick_from(name_i)
                    if cand:
                        picked = cand
                        break
                if picked is not None:
                    return picked
                if prefer_strict:
                    raise ValueError(f"prefer_paper index invalid or methods incomplete: {prefer_paper}")
            elif isinstance(prefer_paper, str):
                cand = _pick_from(prefer_paper)
                if cand:
                    return cand
                if prefer_strict:
                    raise ValueError(f"prefer_paper invalid or methods incomplete: {prefer_paper}")
            # Otherwise continue with voting fallback

        # If topk is empty, direct fallback
        if not topk_papers:
            return {"preprocessing": [], "features": ["pca_feature_extraction"]}

        # ---------- Pairwise (pipeline) majority voting ----------
        def _valid_pair(paper: str) -> Tuple[List[str], List[str], bool]:
            mm = methods_map[paper]
            pre = [x for x in mm.get("preprocessing", []) if x in PRE_FUNCS]
            feat = [x for x in mm.get("features", []) if x in FEAT_FUNCS]
            ok = (len(pre) > 0 and len(feat) > 0) if require_both else (len(pre) > 0 or len(feat) > 0)
            return pre, feat, ok

        pair_counter = Counter()
        key2pair: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[List[str], List[str]]] = {}

        for p in topk_papers:
            pre, feat, ok = _valid_pair(p)
            if not ok:
                continue
            key = _canon_pipeline(pre, feat)
            pair_counter.update([key])
            # Save original lists (maintain order)
            key2pair[key] = (pre, feat)

        chosen: Optional[Dict[str, List[str]]] = None
        if pair_counter:
            mc = pair_counter.most_common()
            top_freq = mc[0][1]
            winners = [k for k, c in mc if c == top_freq]
            if len(winners) == 1 and top_freq >= min_support:
                pre, feat = key2pair[winners[0]]
                chosen = {"preprocessing": pre, "features": feat}

        if chosen:
            return chosen

        # ---------- Tie/no votes: fall back to first complete paper in top K ----------
        fallback_paper = self._first_valid_paper(methods_map, topk_papers, PRE_FUNCS, FEAT_FUNCS, require_both=require_both)
        if fallback_paper is not None:
            sel = methods_map[fallback_paper]
            return {
                "preprocessing": [fn for fn in sel.get("preprocessing", []) if fn in PRE_FUNCS],
                "features": [fn for fn in sel.get("features", []) if fn in FEAT_FUNCS] or ["pca_feature_extraction"],
            }

        # ---------- Still none: final fallback ----------
        return {"preprocessing": [], "features": ["pca_feature_extraction"]}

    def process_with_methods(
        self,
        data: np.ndarray,
        selected: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Process and extract features using the selected set of (paired) methods."""
        PRE_MAP = {fn: getattr(Preprocess, fn) for fn in self.ALL_PREPROCESS_FUNCS}
        FEAT_MAP = {fn: getattr(FeatureExt, fn) for fn in self.ALL_FEATURE_FUNCS}

        pre_list = [fn for fn in selected.get("preprocessing", []) if fn in PRE_MAP]
        feat_list = [fn for fn in selected.get("features", []) if fn in FEAT_MAP]
        if not feat_list:
            feat_list = ["pca_feature_extraction"]

        processed = data.copy()
        for fn in pre_list:
            processed = PRE_MAP[fn](processed)

        feats_dict: Dict[str, np.ndarray] = {}
        for fn in feat_list:
            if fn == "lambert_pearson_feature_extraction":
                if self.regression_label is None:
                    raise ValueError("lambert_pearson_feature_extraction requires regression_label")
                target = self.regression_label
                feats_dict[fn] = FEAT_MAP[fn](processed, target)
            elif fn == "Partial_Least_Squares":
                target = self.regression_label.reshape([-1])
                processed2 = processed.reshape((processed.shape[1], processed.shape[2]))
                feats_dict[fn] = FEAT_MAP[fn](processed2, target)
            else:
                feats_dict[fn] = FEAT_MAP[fn](processed)
        return processed, feats_dict