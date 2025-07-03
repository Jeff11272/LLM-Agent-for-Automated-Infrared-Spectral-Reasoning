
import os
import json
import re
import numpy as np
from openai import OpenAI

# 导入处理模块
import Preprocess_method as Preprocess
import Feature_extract as FeatureExt

# 初始化 Qwen-Plus 客户端
client = OpenAI(
    api_key="...",
    base_url="...",
)

# 可用函数列表：新增 snv_fd, sg_fd, msc_fd
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
    "lambert_pearson_feature_extraction"
]

SYSTEM_PROMPT = f"""
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
  {ALL_PREPROCESS_FUNCS}
Available feature-extraction function names:
  {ALL_FEATURE_FUNCS}

Rules:
 1. Map keywords in descriptions to exact function names.
 2. Do not invent names not in the lists.
 3. Output only valid JSON and nothing else.
"""

def decide_methods_per_paper(papers_info: list) -> dict:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": "papers_info:\n" + json.dumps(papers_info, ensure_ascii=False, indent=2)}
    ]
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages
    )
    text = completion.choices[0].message.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
    # 提取 JSON
    start = text.find('{')
    if start == -1:
        raise RuntimeError(f"No JSON object found in response: {text}")
    depth = 0
    end = None
    for idx, ch in enumerate(text[start:]):
        if   ch == '{': depth += 1
        elif ch == '}': depth -= 1
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

def process_all_papers(data: np.ndarray, methods_map: dict):
    PRE_MAP = {fn: getattr(Preprocess, fn) for fn in ALL_PREPROCESS_FUNCS}
    FEAT_MAP = {fn: getattr(FeatureExt, fn) for fn in ALL_FEATURE_FUNCS}

    # 收集全局可用方法
    global_pre = []
    global_feat = []
    for cfg in methods_map.values():
        valid_pre  = [fn for fn in cfg.get("preprocessing", []) if fn in PRE_MAP]
        valid_feat = [fn for fn in cfg.get("features", []) if fn in FEAT_MAP]
        global_pre.extend(valid_pre)
        global_feat.extend(valid_feat)
    global_pre  = list(dict.fromkeys(global_pre))
    global_feat = list(dict.fromkeys(global_feat))

    # 回退方案
    fallback_pre  = global_pre or []
    fallback_feat = global_feat or ["pca_feature_extraction"]

    result_map = {}
    for paper, cfg in methods_map.items():
        valid_pre  = [fn for fn in cfg.get("preprocessing", []) if fn in PRE_MAP]
        valid_feat = [fn for fn in cfg.get("features", []) if fn in FEAT_MAP]
        pre_list  = valid_pre  or fallback_pre
        feat_list = valid_feat or fallback_feat

        processed = data.copy()
        # 依次应用所有预处理方法
        for fn in pre_list:
            processed = PRE_MAP[fn](processed)

        feats_dict = {}
        # 提取特征
        for fn in feat_list:
            if fn == "lambert_pearson_feature_extraction":
                target = np.load(r'F:\Code\LLM_sp\data\H2O\H2Olabel_js.npy')
                feats_dict[fn] = FEAT_MAP[fn](processed, target)
            else:
                feats_dict[fn] = FEAT_MAP[fn](processed)

        result_map[paper] = {
            "processed_data": processed,
            "extracted_features": feats_dict
        }

    return result_map
