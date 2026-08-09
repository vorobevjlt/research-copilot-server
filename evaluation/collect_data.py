"""
RAGAS Data Collection Script
Runs test questions through your RAG system and collects evaluation data.
"""

import json
from pathlib import Path
import sys
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retrieval.index import retrieve_context
from src.rag.retrieval.utils import prepare_prompt_and_invoke_llm

# Configuration
PROJECT_ID = "9bfdd5e5-2504-493e-bf1c-d179d52ceed0"

TEST_QUESTIONS = [
    # "What is the Big Bang theory?",
    # "How many neurons does the human brain contain?",
    "How does the Transformer architecture differ from recurrent and convolutional sequence-transduction models?",
    "How does scaled dot-product attention work, and why are the dot products divided by \(\sqrt{d_k}\)?",
    "What is the purpose of multi-head attention, and how is it applied in the Transformer’s encoder and decoder?",
    "Why does the Transformer require positional encodings, and why did the authors choose sinusoidal functions?",
    "What BLEU scores and training costs did the Transformer achieve on the WMT 2014 English-to-German and English-to-French translation tasks?",
    "What is the overall architecture of the convolutional neural network described in the paper?",
    "Why did the authors use ReLU neurons instead of saturating nonlinearities such as tanh?",
    "How was the network divided between two GPUs, and what benefits did this provide?",
    "How did data augmentation and dropout help reduce overfitting in the network?",
    "What top-1 and top-5 error rates did the model achieve on the ILSVRC-2010 and ILSVRC-2012 datasets?",
]


def collect_rag_data(project_id: str, questions: list) -> list:
    """Run questions through RAG pipeline and collect data."""
    dataset = []

    for question in questions:
        print(f"Processing: {question}")

        # Retrieve context
        texts, images, tables, citations = retrieve_context(project_id, question)

        # Prepare contexts for RAGAS
        contexts = texts + [f"[TABLE]\n{table}" for table in tables]

        # Generate answer
        answer = prepare_prompt_and_invoke_llm(question, texts, [], tables)

        dataset.append({
            "question": question,
            "contexts": contexts or ["No context found"],
            "answer": answer
        })

    return dataset


def append_to_dataset(output_path: Path, new_records: list) -> int:
    """Append records to an existing JSON dataset and return its new size."""
    existing_records = []

    if output_path.exists():
        existing_content = output_path.read_text(encoding="utf-8").strip()
        if existing_content:
            existing_records = json.loads(existing_content)

        if not isinstance(existing_records, list):
            raise ValueError(f"Expected a JSON list in {output_path}")

    existing_records.extend(new_records)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(existing_records, f, indent=2, ensure_ascii=False)

    return len(existing_records)


if __name__ == "__main__":
    # Collect and save data
    dataset = collect_rag_data(PROJECT_ID, TEST_QUESTIONS)

    output_path = Path(__file__).parent / "datasets" / "ragas_evaluation_dataset-1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = append_to_dataset(output_path, dataset)

    print(
        f"\n✅ Added {len(dataset)} questions to {output_path} "
        f"({total_records} total)"
    )
