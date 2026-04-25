#!/usr/bin/env python3
"""
Batch hypergraph construction from all Markdown files in a directory.

All settings are read from a YAML configuration file (default: config.yaml).
Any individual value can be overridden at runtime with a CLI flag.

Usage:
  # Use the default config.yaml
  python script_batch_construct.py

  # Point at a different config file
  python script_batch_construct.py --config my_config.yaml

  # Override individual values without editing the file
  python script_batch_construct.py --llm-model Qwen/Qwen2.5-7B-Instruct

Run with --help to see all available flags.
"""

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import yaml

from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import openai_complete_if_cache, openai_embedding
from hypergraphrag.utils import EmbeddingFunc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded fallbacks used when a key is absent from both YAML and CLI.
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "data_dir": "./data",
    "working_dir": "./expr/batch",
    "llm_base_url": "http://localhost:8000/v1",
    "llm_api_key": "EMPTY",
    "llm_model": "",
    "llm_max_token_size": 32768,
    "llm_max_async": 4,
    "embed_base_url": "http://localhost:8000/v1",
    "embed_api_key": "EMPTY",
    "embed_model": "",
    "embed_dim": 4096,
    "embed_max_tokens": 8192,
    "embed_max_async": 8,
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "entity_extract_max_gleaning": 2,
    "entity_summary_to_max_tokens": 500,
    "enable_llm_cache": True,
    "max_retries": 5,
}


def _load_yaml(path: str) -> dict:
    """Load YAML config and flatten nested sections into a single dict."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Config file '%s' not found — using defaults.", path)
        return {}
    except yaml.YAMLError as exc:
        logger.error("Failed to parse '%s': %s", path, exc)
        sys.exit(1)

    paths = raw.get("paths", {})
    llm = raw.get("llm", {})
    emb = raw.get("embedding", {})
    graph = raw.get("graph", {})
    batch = raw.get("batch", {})

    return {
        "data_dir": paths.get("data_dir"),
        "working_dir": paths.get("working_dir"),
        "llm_base_url": llm.get("base_url"),
        "llm_api_key": llm.get("api_key"),
        "llm_model": llm.get("model"),
        "llm_max_token_size": llm.get("max_token_size"),
        "llm_max_async": llm.get("max_async"),
        "embed_base_url": emb.get("base_url"),
        "embed_api_key": emb.get("api_key"),
        "embed_model": emb.get("model"),
        "embed_dim": emb.get("dim"),
        "embed_max_tokens": emb.get("max_tokens"),
        "embed_max_async": emb.get("max_async"),
        "chunk_token_size": graph.get("chunk_token_size"),
        "chunk_overlap_token_size": graph.get("chunk_overlap_token_size"),
        "entity_extract_max_gleaning": graph.get("entity_extract_max_gleaning"),
        "entity_summary_to_max_tokens": graph.get("entity_summary_to_max_tokens"),
        "enable_llm_cache": graph.get("enable_llm_cache"),
        "max_retries": batch.get("max_retries"),
    }


def parse_args() -> argparse.Namespace:
    """
    Two-pass argument parsing so that --config is resolved before building
    the full parser, allowing YAML values to serve as argparse defaults.
    Priority: CLI flag > YAML value > hard-coded default.
    """
    # Pass 1 – resolve config file path only.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()

    # Load YAML; merge with hard-coded defaults (YAML wins over _DEFAULTS).
    yaml_cfg = _load_yaml(pre_args.config)
    merged = {k: (yaml_cfg[k] if yaml_cfg.get(k) is not None else v)
              for k, v in _DEFAULTS.items()}

    # Pass 2 – full parser; YAML-merged values become argparse defaults so any
    # explicitly supplied CLI flag still takes precedence.
    parser = argparse.ArgumentParser(
        description="Build a HyperGraphRAG hypergraph from Markdown files "
        "using a locally deployed vLLM instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to the YAML configuration file.",
    )

    # -- Paths --
    parser.add_argument("--data-dir", default=merged["data_dir"],
                        help="Directory containing .md / .markdown files.")
    parser.add_argument("--working-dir", default=merged["working_dir"],
                        help="Output directory for the hypergraph artefacts.")

    # -- LLM --
    parser.add_argument("--llm-base-url", default=merged["llm_base_url"],
                        help="OpenAI-compatible endpoint for the LLM server.")
    parser.add_argument("--llm-api-key", default=merged["llm_api_key"],
                        help="API key for the LLM server.")
    parser.add_argument("--llm-model", default=merged["llm_model"],
                        help="LLM model name as served by the LLM server.")
    parser.add_argument("--llm-max-token-size", type=int,
                        default=merged["llm_max_token_size"],
                        help="Maximum tokens the LLM may generate per call.")
    parser.add_argument("--llm-max-async", type=int,
                        default=merged["llm_max_async"],
                        help="Maximum concurrent LLM requests.")

    # -- Embedding --
    parser.add_argument("--embed-base-url", default=merged["embed_base_url"],
                        help="OpenAI-compatible endpoint for the embedding server.")
    parser.add_argument("--embed-api-key", default=merged["embed_api_key"],
                        help="API key for the embedding server.")
    parser.add_argument("--embed-model", default=merged["embed_model"],
                        help="Embedding model name as served by the embedding server.")
    parser.add_argument("--embed-dim", type=int, default=merged["embed_dim"],
                        help="Output vector dimension of the embedding model.")
    parser.add_argument("--embed-max-tokens", type=int,
                        default=merged["embed_max_tokens"],
                        help="Maximum input tokens the embedding model accepts.")
    parser.add_argument("--embed-max-async", type=int,
                        default=merged["embed_max_async"],
                        help="Maximum concurrent embedding requests.")

    # -- Graph construction --
    parser.add_argument("--chunk-token-size", type=int,
                        default=merged["chunk_token_size"],
                        help="Target token size for each document chunk.")
    parser.add_argument("--chunk-overlap-token-size", type=int,
                        default=merged["chunk_overlap_token_size"],
                        help="Token overlap between adjacent chunks.")
    parser.add_argument("--entity-extract-max-gleaning", type=int,
                        default=merged["entity_extract_max_gleaning"],
                        help="Extra LLM re-prompts per chunk for entity extraction.")
    parser.add_argument("--entity-summary-to-max-tokens", type=int,
                        default=merged["entity_summary_to_max_tokens"],
                        help="Max tokens for summarising an entity description.")
    parser.add_argument("--enable-llm-cache", type=_parse_bool,
                        default=merged["enable_llm_cache"],
                        metavar="BOOL",
                        help="Cache LLM responses to disk (true/false).")

    # -- Batch --
    parser.add_argument("--max-retries", type=int, default=merged["max_retries"],
                        help="Insertion retry attempts before giving up.")

    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    """Allow --enable-llm-cache true/false/1/0/yes/no on the CLI."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value!r}")


def collect_markdown_files(data_dir: str) -> list:
    """Return a sorted list of all .md and .markdown files under data_dir."""
    files = []
    for pattern in ("**/*.md", "**/*.markdown"):
        files.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
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
    """Construct a HyperGraphRAG instance wired to the configured model servers."""
    llm_base_url: str = args.llm_base_url
    llm_api_key: str = args.llm_api_key
    llm_model: str = args.llm_model
    embed_base_url: str = args.embed_base_url
    embed_api_key: str = args.embed_api_key
    embed_model: str = args.embed_model

    async def vllm_llm_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("keyword_extraction", None)
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=llm_base_url,
            api_key=llm_api_key,
            **kwargs,
        )

    # Call the underlying async embedding function directly so we can supply
    # a custom EmbeddingFunc with the correct dim / max_tokens for this model.
    async def vllm_embed_func(texts: list) -> np.ndarray:
        return await openai_embedding.func(
            texts,
            model=embed_model,
            base_url=embed_base_url,
            api_key=embed_api_key,
        )

    # concurrent_limit=0 disables EmbeddingFunc's own semaphore; HyperGraphRAG
    # applies its own rate-limit via embedding_func_max_async.
    embedding_func = EmbeddingFunc(
        embedding_dim=args.embed_dim,
        max_token_size=args.embed_max_tokens,
        func=vllm_embed_func,
        concurrent_limit=0,
    )

    os.makedirs(args.working_dir, exist_ok=True)

    return HyperGraphRAG(
        working_dir=args.working_dir,
        # LLM
        llm_model_func=vllm_llm_func,
        llm_model_name=llm_model,
        llm_model_max_token_size=args.llm_max_token_size,
        llm_model_max_async=args.llm_max_async,
        # Embedding
        embedding_func=embedding_func,
        embedding_func_max_async=args.embed_max_async,
        # Graph construction
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
        entity_extract_max_gleaning=args.entity_extract_max_gleaning,
        entity_summary_to_max_tokens=args.entity_summary_to_max_tokens,
        enable_llm_cache=args.enable_llm_cache,
    )


def insert_with_retry(rag: HyperGraphRAG, texts: list, max_retries: int) -> bool:
    """Insert documents into the graph, retrying with exponential back-off."""
    for attempt in range(1, max_retries + 1):
        try:
            rag.insert(texts)
            return True
        except Exception as exc:
            wait = 2 ** attempt
            if attempt < max_retries:
                logger.warning(
                    "Insertion attempt %d/%d failed: %s — retrying in %ds",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Insertion attempt %d/%d failed: %s — no more retries.",
                    attempt, max_retries, exc,
                )
    return False


def main() -> None:
    args = parse_args()

    # The OpenAI client library requires OPENAI_API_KEY to be set.
    os.environ.setdefault("OPENAI_API_KEY", args.llm_api_key)

    logger.info("Config file    : %s", args.config)
    logger.info("LLM base URL   : %s", args.llm_base_url)
    logger.info("LLM model      : %s", args.llm_model or "(server default)")
    logger.info("Embed base URL : %s", args.embed_base_url)
    logger.info("Embed model    : %s", args.embed_model or "(server default)")
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
        logger.error(
            "Hypergraph construction failed after %d attempts.", args.max_retries
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
