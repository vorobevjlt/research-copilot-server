"""
Simple RAGAS Evaluation Script
"""
import pandas as pd
import json
from pathlib import Path
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.services.llm import openAI
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper

load_dotenv()

# Load your dataset
path = Path(__file__).parent / "datasets" / "ragas_evaluation_dataset-1.json"
with path.open(encoding="utf-8") as f:
    data = json.load(f)


# Convert to RAGAS format
dataset = Dataset.from_dict({
    "question": [item["question"] for item in data],
    "answer": [item["answer"] for item in data],
    "contexts": [item["contexts"] for item in data],
})

# Set up evaluator (using GPT-4 for evaluation)
llm = openAI["chat_llm"]
embeddings = openAI["embeddings"]
evaluator_llm = LangchainLLMWrapper(
    openAI["chat_llm"],
    bypass_temperature=True,
    bypass_n=True,
)
# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ],
    raise_exceptions=True,
    llm=evaluator_llm,
    embeddings=embeddings,
)

# Convert to DataFrame first
df = results.to_pandas()
print(df.head())
# Save to CSV
# output_path = Path(__file__).parent / "datasets" / "results.csv"
# output_path.parent.mkdir(parents=True, exist_ok=True)
# df.to_csv(output_path, index=False)
