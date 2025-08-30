import json
import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk

# If you haven't downloaded the punkt tokenizer yet, uncomment and run the following line once:
# nltk.download('punkt')

def load_papers_from_json(json_path: str) -> List[Dict]:
    """
    Load JSON file from specified path and return a list of paper entries.

    Parameters:
        json_path (str): Absolute or relative path to the JSON file. The file content should be a JSON array,
                         with each element formatted as:
                         {
                           "paper_name": "...",
                           "paper_url": "...",
                           "research_object": "...",
                           "preprocessing_method": "...",
                           "feature_extracting_method": "...",
                           "machine_learning_method": "..."
                         }
    Returns:
        List[Dict]: A list of dictionaries corresponding to each paper in the JSON.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Cannot find the specified JSON file: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The JSON file content should be an array (list), with each element being a dictionary containing paper information.")
    return data


def build_bm25_index(papers: List[Dict]) -> (BM25Okapi, List[List[str]]):
    """
    Build an index using BM25, with indexing content derived from tokenized "paper_name" fields of each paper.

    Parameters:
        papers (List[Dict]): List of papers, each element contains fields like "paper_name".

    Returns:
        tuple:
            - BM25Okapi: BM25 indexer built based on all paper_name texts.
            - List[List[str]]: Tokenized list of all original paper_names, maintaining consistent order with the BM25 indexer.
    """
    # Tokenize each paper_name (using nltk.word_tokenize here)
    tokenized_paper_names: List[List[str]] = []
    for entry in papers:
        name: str = entry.get("paper_name", "")
        research_object: str = entry.get("research_object", "")
        combined_text = f"{name} {research_object}".strip().lower()
        tokens = nltk.word_tokenize(combined_text.lower())
        tokenized_paper_names.append(tokens)
    # Build BM25 index using tokenized results
    bm25 = BM25Okapi(tokenized_paper_names)
    return bm25, tokenized_paper_names


def search_papers_with_bm25(
    papers: List[Dict],
    bm25: BM25Okapi,
    tokenized_paper_names: List[List[str]],
    query: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Retrieve top_k most relevant papers using BM25 for the query (research object entity), 
    and return their preprocessing and feature extraction methods.

    Parameters:
        papers (List[Dict]): List of papers, each element contains fields such as "paper_name", "preprocessing_method", "feature_extracting_method".
        bm25 (BM25Okapi): Pre-built BM25 indexer corresponding to tokenized_paper_names.
        tokenized_paper_names (List[List[str]]): Tokenized list of paper_names corresponding to the bm25 indexer.
        query (str): The "research object" entity extracted by LLM, such as "ink", "stamp ink", etc.
        top_k (int): Number of most relevant papers to return.

    Returns:
        List[Dict]: A list of dictionaries sorted by BM25 scores in descending order, each dictionary contains:
            - paper_name: Paper name
            - preprocessing_method: Preprocessing method of that paper
            - feature_extracting_method: Feature extraction method of that paper
            - score: BM25 relevance score
    """
    # Tokenize the query
    query_tokens = nltk.word_tokenize(query.lower())

    # Calculate BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Sort (index, score) pairs by score in descending order
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