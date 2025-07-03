import json
import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk

def load_papers_from_json(json_path: str) -> List[Dict]:

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"找不到指定的 JSON 文件：{json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件内容应当是一个数组（list），每个元素为一篇论文的字典信息。")
    return data


def build_bm25_index(papers: List[Dict]) -> (BM25Okapi, List[List[str]]):

    tokenized_paper_names: List[List[str]] = []
    for entry in papers:
        name: str = entry.get("paper_name", "")
        # 转为小写并分词
        tokens = nltk.word_tokenize(name.lower())
        tokenized_paper_names.append(tokens)

    # 用分词结果构建 BM25 索引
    bm25 = BM25Okapi(tokenized_paper_names)
    return bm25, tokenized_paper_names


def search_papers_with_bm25(
    papers: List[Dict],
    bm25: BM25Okapi,
    tokenized_paper_names: List[List[str]],
    query: str,
    top_k: int = 3
) -> List[Dict]:

    query_tokens = nltk.word_tokenize(query.lower())

    # 计算 BM25 得分
    scores = bm25.get_scores(query_tokens)

    # 将 (索引, 得分) 对按得分降序排序
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results: List[Dict] = []
    for idx, score in ranked[:top_k]:
        entry = papers[idx]
        results.append({
            "paper_name": entry.get("paper_name", ""),
            "preprocessing_method": entry.get("best_preprocessing_method", ""),
            "feature_extracting_method": entry.get("best_feature_extracting_method", ""),
            #"score": float(score)
        })
        print('Paper_Name:',entry.get("paper_name", ""))
        print('Relevant Scores:', score)
        
    return results