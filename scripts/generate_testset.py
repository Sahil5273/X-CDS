"""Generate a synthetic clinical evaluation dataset from local BioC JSONL passages using Ragas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import sys
from types import ModuleType
import langchain_google_genai

# Mock langchain_community.chat_models.vertexai for Ragas compatibility
if "langchain_community.chat_models.vertexai" not in sys.modules:
    mock_module = ModuleType("langchain_community.chat_models.vertexai")
    mock_module.ChatVertexAI = langchain_google_genai.ChatGoogleGenerativeAI
    sys.modules["langchain_community.chat_models.vertexai"] = mock_module

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

# Load local environment parameters
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/bioc_chunks.jsonl"),
        help="Path to the BioC JSONL knowledge base passages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/my_eval_set_large.jsonl"),
        help="Path to write the generated JSONL dataset.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="Number of clinical evaluation cases to generate.",
    )
    return parser


def load_bioc_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise SystemExit(f"Input JSONL does not exist: {path}")

    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Create a LangChain Document representation
            docs.append(
                Document(
                    page_content=record["text"],
                    metadata={
                        "chunk_id": record["chunk_id"],
                        "pmcid": record["pmcid"],
                        "section": record["section"],
                    },
                )
            )
    return docs


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading documents from {args.input}...")
    documents = load_bioc_documents(args.input)
    print(f"Loaded {len(documents)} document chunks.")

    # Initialize Gemini models via langchain-google-genai using Vertex AI credentials
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "global")

    print(
        f"Initializing Gemini models (Project: {project_id}, Region: {region})...."
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        vertexai=True,
        project=project_id,
        location=region,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        vertexai=True,
        project=project_id,
        location=region,
    )

    # Wrap the models for Ragas TestsetGenerator
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    print("Initializing Ragas TestsetGenerator...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    print(f"Generating {args.size} synthetic clinical evaluation cases...")
    # Generate the test set
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=args.size,
    )

    df = testset.to_pandas()

    print(f"Writing dataset to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for _, row in df.iterrows():
            record = {
                "question": str(row["question"]),
                "ground_truth": str(row["ground_truth"]),
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(df)} test cases to {args.output}")


if __name__ == "__main__":
    main()
