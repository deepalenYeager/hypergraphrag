#!/usr/bin/env python3
"""
Query a HyperGraphRAG hypergraph using a locally deployed vLLM instance.

All settings are read from a YAML configuration file (default: config.yaml).
Any individual value can be overridden at runtime with a CLI flag.

Usage:
  # Query text supplied directly
  python script_query.py "What is hypertension?"

  # Use a different config file
  python script_query.py --config my_config.yaml "What is hypertension?"

  # Override retrieval mode for a single run
  python script_query.py --mode local "What is hypertension?"

Run with --help to see all available flags.
"""

import argparse
import logging
import os
import sys

import numpy as np
import yaml

from hypergraphrag import HyperGraphRAG, QueryParam
from hypergraphrag.llm import openai_complete_if_cache, openai_embedding
from hypergraphrag.utils import EmbeddingFunc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULTS = {
    "working_dir": "./expr/batch",
    "vllm_base_url": "http://localhost:8000/v1",
    "llm_model": "",
    "llm_max_token_size": 32768,
    "llm_max_async": 4,
    "embed_model": "",
    "embed_dim": 4096,
    "embed_max_tokens": 8192,
    "embed_max_async": 8,
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "entity_extract_max_gleaning": 2,
    "entity_summary_to_max_tokens": 500,
    "enable_llm_cache": True,
    "query_mode": "hybrid",
    "top_k": 60,
    "response_type": "Multiple Paragraphs",
    "max_token_for_text_unit": 4000,
    "max_token_for_global_context": 4000,
    "max_token_for_local_context": 4000,
}


def _load_yaml(path: str) -> dict:
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
    vllm = raw.get("vllm", {})
    llm = raw.get("llm", {})
    emb = raw.get("embedding", {})
    graph = raw.get("graph", {})
    q = raw.get("query", {})

    return {
        "working_dir": paths.get("working_dir"),
        "vllm_base_url": vllm.get("base_url"),
        "llm_model": llm.get("model"),
        "llm_max_token_size": llm.get("max_token_size"),
        "llm_max_async": llm.get("max_async"),
        "embed_model": emb.get("model"),
        "embed_dim": emb.get("dim"),
        "embed_max_tokens": emb.get("max_tokens"),
        "embed_max_async": emb.get("max_async"),
        "chunk_token_size": graph.get("chunk_token_size"),
        "chunk_overlap_token_size": graph.get("chunk_overlap_token_size"),
        "entity_extract_max_gleaning": graph.get("entity_extract_max_gleaning"),
        "entity_summary_to_max_tokens": graph.get("entity_summary_to_max_tokens"),
        "enable_llm_cache": graph.get("enable_llm_cache"),
        "query_mode": q.get("mode"),
        "top_k": q.get("top_k"),
        "response_type": q.get("response_type"),
        "max_token_for_text_unit": q.get("max_token_for_text_unit"),
        "max_token_for_global_context": q.get("max_token_for_global_context"),
        "max_token_for_local_context": q.get("max_token_for_local_context"),
    }


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value!r}")


def parse_args() -> argparse.Namespace:
    # Pass 1 – resolve config file path only.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    pre_args, _ = pre.parse_known_args()

    yaml_cfg = _load_yaml(pre_args.config)
    merged = {k: (yaml_cfg[k] if yaml_cfg.get(k) is not None else v)
              for k, v in _DEFAULTS.items()}

    # Pass 2 – full parser with YAML-merged defaults.
    parser = argparse.ArgumentParser(
        description="Query a HyperGraphRAG hypergraph using a locally deployed vLLM instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml", metavar="FILE",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "query_text", nargs="?", default=None,
        help="Query string. If omitted, the script reads from stdin.",
    )

    # -- RAG instance --
    parser.add_argument("--working-dir", default=merged["working_dir"],
                        help="Directory containing the pre-built hypergraph artefacts.")
    parser.add_argument("--vllm-base-url", default=merged["vllm_base_url"],
                        help="Base URL of the vLLM OpenAI-compatible server.")

    # -- LLM --
    parser.add_argument("--llm-model", default=merged["llm_model"],
                        help="LLM model name as served by vLLM.")
    parser.add_argument("--llm-max-token-size", type=int,
                        default=merged["llm_max_token_size"],
                        help="Maximum tokens the LLM may generate per call.")
    parser.add_argument("--llm-max-async", type=int, default=merged["llm_max_async"],
                        help="Maximum concurrent LLM requests.")

    # -- Embedding --
    parser.add_argument("--embed-model", default=merged["embed_model"],
                        help="Embedding model name as served by vLLM.")
    parser.add_argument("--embed-dim", type=int, default=merged["embed_dim"],
                        help="Output vector dimension of the embedding model.")
    parser.add_argument("--embed-max-tokens", type=int,
                        default=merged["embed_max_tokens"],
                        help="Maximum input tokens the embedding model accepts.")
    parser.add_argument("--embed-max-async", type=int,
                        default=merged["embed_max_async"],
                        help="Maximum concurrent embedding requests.")

    # -- Graph construction (must match values used during insert) --
    parser.add_argument("--chunk-token-size", type=int,
                        default=merged["chunk_token_size"])
    parser.add_argument("--chunk-overlap-token-size", type=int,
                        default=merged["chunk_overlap_token_size"])
    parser.add_argument("--entity-extract-max-gleaning", type=int,
                        default=merged["entity_extract_max_gleaning"])
    parser.add_argument("--entity-summary-to-max-tokens", type=int,
                        default=merged["entity_summary_to_max_tokens"])
    parser.add_argument("--enable-llm-cache", type=_parse_bool,
                        default=merged["enable_llm_cache"], metavar="BOOL",
                        help="Cache LLM responses to disk (true/false).")

    # -- Query behaviour --
    parser.add_argument(
        "--mode", dest="query_mode",
        choices=["local", "global", "hybrid", "naive"],
        default=merged["query_mode"],
        help="Retrieval mode.",
    )
    parser.add_argument("--top-k", type=int, default=merged["top_k"],
                        help="Number of top-k entities / relationships to retrieve.")
    parser.add_argument("--response-type", default=merged["response_type"],
                        help="Format hint for the LLM answer (e.g. 'Multiple Paragraphs').")
    parser.add_argument("--max-token-for-text-unit", type=int,
                        default=merged["max_token_for_text_unit"],
                        help="Token budget for retrieved text chunks.")
    parser.add_argument("--max-token-for-global-context", type=int,
                        default=merged["max_token_for_global_context"],
                        help="Token budget for the global (hyperedge) context.")
    parser.add_argument("--max-token-for-local-context", type=int,
                        default=merged["max_token_for_local_context"],
                        help="Token budget for the local (entity) context.")

    return parser.parse_args()


def build_rag(args: argparse.Namespace) -> HyperGraphRAG:
    base_url: str = args.vllm_base_url
    llm_model: str = args.llm_model
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
            base_url=base_url,
            api_key="EMPTY",
            **kwargs,
        )

    async def vllm_embed_func(texts: list) -> np.ndarray:
        return await openai_embedding.func(
            texts,
            model=embed_model,
            base_url=base_url,
            api_key="EMPTY",
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=args.embed_dim,
        max_token_size=args.embed_max_tokens,
        func=vllm_embed_func,
        concurrent_limit=0,
    )

    return HyperGraphRAG(
        working_dir=args.working_dir,
        llm_model_func=vllm_llm_func,
        llm_model_name=llm_model,
        llm_model_max_token_size=args.llm_max_token_size,
        llm_model_max_async=args.llm_max_async,
        embedding_func=embedding_func,
        embedding_func_max_async=args.embed_max_async,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
        entity_extract_max_gleaning=args.entity_extract_max_gleaning,
        entity_summary_to_max_tokens=args.entity_summary_to_max_tokens,
        enable_llm_cache=args.enable_llm_cache,
    )


def main() -> None:
    args = parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    # Resolve query text: CLI arg → stdin.
    query_text = args.query_text
    if not query_text:
        if sys.stdin.isatty():
            print("Enter query: ", end="", flush=True)
        query_text = sys.stdin.read().strip()
    if not query_text:
        logger.error("No query text provided. Exiting.")
        sys.exit(1)

    logger.info("Config file    : %s", args.config)
    logger.info("Working dir    : %s", args.working_dir)
    logger.info("vLLM base URL  : %s", args.vllm_base_url)
    logger.info("LLM model      : %s", args.llm_model or "(server default)")
    logger.info("Mode           : %s", args.query_mode)
    logger.info("Top-k          : %d", args.top_k)

    rag = build_rag(args)

    param = QueryParam(
        mode=args.query_mode,
        top_k=args.top_k,
        response_type=args.response_type,
        max_token_for_text_unit=args.max_token_for_text_unit,
        max_token_for_global_context=args.max_token_for_global_context,
        max_token_for_local_context=args.max_token_for_local_context,
    )

    result = rag.query(query_text, param=param)
    print(result)


if __name__ == "__main__":
    main()
