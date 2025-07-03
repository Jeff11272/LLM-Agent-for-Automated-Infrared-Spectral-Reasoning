import os
from openai import OpenAI
import json
def extract_entities_and_task(input_text: str) -> dict:
    """
    Uses an LLM to extract both the research object entity and the task type
    (one of: 'classification', 'regression', 'anomaly detection') from the user's question.

    Returns:
        dict with keys:
            - "research_object": str
            - "task_type": str  # one of "classification", "regression", "anomaly detection"
    """
    client = OpenAI(
        api_key="...",
        base_url="...",
    )

    SYSTEM_PROMPT = """
You are a specialized assistant for extracting two pieces of information from a user's question:
1. The single research object entity (e.g., material, sample type, analyte).
2. The task type the user intends, which must be exactly one of:
   - "classification"
   - "regression"
   - "anomaly detection"

Output ONLY a valid JSON dictionary with exactly two keys: "research_object" and "task_type".
Do NOT wrap your output in markdown code blocks or any other formatting.
Do NOT include any additional text or explanation.

Example:
{"research_object": "semiconductor material", "task_type": "classification"}
"""

    user_prompt = f"""
Please analyze the following question and extract:
1. The research object entity.
2. Which task type it is (classification, regression, or anomaly detection).

Question:
\"\"\"{input_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    
    content = response.choices[0].message.content.strip()
    try:
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        raise ValueError(f"Unexpected LLM output: {content}")
