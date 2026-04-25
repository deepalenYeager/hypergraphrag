#!/usr/bin/env python3
"""
Batch hypergraph construction from all Markdown files in a directory.

Uses a locally deployed vLLM instance (OpenAI-compatible API) for both
the language model and the embedding model.

Configuration can be supplied via CLI flags or environment variables:
  --vllm-base-url / VLLM_BASE_URL     vLLM server URL (default: http://localhost:8000/v1)
  --llm-model     / VLLM_LLM_MODEL    Model name served as the LLM
  --embed-model   / VLLM_EMBED_MODEL  Model name served for embeddings
  --embed-dim     / VLLM_EMBED_DIM    Embedding vector dimension (default: 4096)
  --embed-max-tokens / VLLM_EMBED_MAX_TOKENS  Max tokens for embedding model (default: 8192)
  --data-dir      / DATA_DIR          Markdown source directory (default: ./data)
  --working-dir   / WORKING_DIR       Graph output directory (default: ./expr/batch)

Usage example:
  python script_batch_construct.py \\
      --vllm-base-url http://localhost:8000/v1 \\
      --llm-model Qwen/Qwen2.5-7B-Instruct \\
      --embed-model BAAI/bge-m3 \\
      --embed-dim 1024 \\
      --data-dir ./data
"""

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np

from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import openai_complete_if_cache, openai_embedding
from hypergraphrag.utils import EmbeddingFunc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a HyperGraphRAG knowledge graph from Markdown files "
        "using a locally deployed vLLM instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "./data"),
        help="Directory containing Markdown (.md / .markdown) files.",
    )
    parser.add_argument(
        "--working-dir",
        default=os.environ.get("WORKING_DIR", "./expr/batch"),
        help="Output directory where the hypergraph is persisted.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="Base URL of the vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("VLLM_LLM_MODEL", ""),
        help="Name of the LLM model served by vLLM (e.g. Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("VLLM_EMBED_MODEL", ""),
        help="Name of the embedding model served by vLLM (e.g. BAAI/bge-m3).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=int(os.environ.get("VLLM_EMBED_DIM", "4096")),
        help="Output dimension of the embedding model.",
    )
    parser.add_argument(
        "--embed-max-tokens",
        type=int,
        default=int(os.environ.get("VLLM_EMBED_MAX_TOKENS", "8192")),
        help="Maximum input token length accepted by the embedding model.",
    )
    parser.add_argument(
        "--llm-max-async",
        type=int,
        default=4,
        help="Maximum number of concurrent LLM requests.",
    )
    parser.add_argument(
        "--embed-max-async",
        type=int,
        default=8,
        help="Maximum number of concurrent embedding requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum insertion retry attempts before giving up.",
    )
    return parser.parse_args()


def collect_markdown_files(data_dir: str) -> list:
    """Return a sorted list of all .md and .markdown files under data_dir."""
    patterns = [
        os.path.join(data_dir, "**", "*.md"),
        os.path.join(data_dir, "**", "*.markdown"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def read_markdown_files(file_paths: list) -> list:
    """Read each file and return a list of non-empty text strings."""
    texts = []
    for path in file_paths:
        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read().strip()
            if content:
                texts.append(content)
                logger.info("Loaded %s (%d chars)", path, len(content))
            else:
                logger.warning("Skipping empty file: %s", path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
    return texts


def build_rag(args: argparse.Namespace) -> HyperGraphRAG:
    """Construct a HyperGraphRAG instance wired to the local vLLM server."""
    base_url: str = args.vllm_base_url
    llm_model: str = args.llm_model
    embed_model: str = args.embed_model

    # vLLM exposes an OpenAI-compatible REST API.  We reuse
    # openai_complete_if_cache by pointing it at the local server.
    async def vllm_llm_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("keyword_extraction", None)
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=base_url,
            api_key="EMPTY",  # vLLM does not validate API keys
            **kwargs,
        )

    # vLLM also serves embedding models via /v1/embeddings.
    # We call the underlying async function directly so we can supply our own
    # EmbeddingFunc wrapper with the correct embedding_dim / max_token_size.
    async def vllm_embed_func(texts: list) -> np.ndarray:
        return await openai_embedding.func(
            texts,
            model=embed_model,
            base_url=base_url,
            api_key="EMPTY",
        )

    # concurrent_limit=0 disables EmbeddingFunc's internal semaphore;
    # HyperGraphRAG will apply its own rate-limit via embedding_func_max_async.
    embedding_func = EmbeddingFunc(
        embedding_dim=args.embed_dim,
        max_token_size=args.embed_max_tokens,
        func=vllm_embed_func,
        concurrent_limit=0,
    )

    os.makedirs(args.working_dir, exist_ok=True)

    return HyperGraphRAG(
        working_dir=args.working_dir,
        llm_model_func=vllm_llm_func,
        llm_model_name=llm_model,
        llm_model_max_async=args.llm_max_async,
        embedding_func=embedding_func,
        embedding_func_max_async=args.embed_max_async,
    )


def insert_with_retry(rag: HyperGraphRAG, texts: list, max_retries: int) -> bool:
    """Insert documents into the graph, retrying with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            rag.insert(texts)
            return True
        except Exception as exc:
            wait = 2 ** attempt
            if attempt < max_retries:
                logger.warning(
                    "Insertion attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Insertion attempt %d/%d failed: %s — no more retries.",
                    attempt,
                    max_retries,
                    exc,
                )
    return False


def main() -> None:
    args = parse_args()

    # The OpenAI client requires OPENAI_API_KEY to be set even when talking to
    # a local vLLM server that does not validate keys.
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    logger.info("vLLM base URL  : %s", args.vllm_base_url)
    logger.info("LLM model      : %s", args.llm_model or "(server default)")
    logger.info("Embedding model: %s", args.embed_model or "(server default)")
    logger.info("Embed dim      : %d", args.embed_dim)
    logger.info("Data directory : %s", args.data_dir)
    logger.info("Working dir    : %s", args.working_dir)

    # --- Collect Markdown documents ---
    md_files = collect_markdown_files(args.data_dir)
    if not md_files:
        logger.error("No Markdown files found in '%s'. Exiting.", args.data_dir)
        sys.exit(1)
    logger.info("Found %d Markdown file(s).", len(md_files))

    texts = read_markdown_files(md_files)
    if not texts:
        logger.error("All Markdown files were empty or unreadable. Exiting.")
        sys.exit(1)
    logger.info("Loaded %d non-empty document(s).", len(texts))

    # --- Initialise RAG instance ---
    rag = build_rag(args)

    # --- Build hypergraph ---
    logger.info("Starting hypergraph construction …")
    success = insert_with_retry(rag, texts, args.max_retries)
    if success:
        logger.info("Hypergraph construction completed successfully.")
    else:
        logger.error("Hypergraph construction failed after %d attempts.", args.max_retries)
        sys.exit(1)


if __name__ == "__main__":
    main()
