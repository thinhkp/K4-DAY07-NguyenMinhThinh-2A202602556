from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Protocol

from dotenv import load_dotenv

from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)


DATA_DIR = Path("data/shopee-return-refund")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500
REQUIRED_METADATA = {
    "doc_id",
    "title",
    "source_url",
    "retrieved_at",
    "document_version",
    "audience",
}

BENCHMARKS = [
    {
        "id": 1,
        "query": "Với thực phẩm tươi sống và đông lạnh, người mua phải gửi yêu cầu Trả hàng/Hoàn tiền trong thời hạn bao lâu?",
        "gold_answer": "Trong vòng 24 giờ kể từ khi đơn hàng được cập nhật trạng thái Giao hàng thành công, trừ trường hợp chưa nhận được hàng.",
        "gold_doc_id": "buyer-return-conditions",
        "answer_markers": ("trong vòng 24 giờ", "thực phẩm tươi sống"),
        "metadata_filter": {"audience": "buyer"},
    },
    {
        "id": 2,
        "query": "Shopee có hỗ trợ yêu cầu đổi hàng không, và người mua có thể làm gì nếu hàng nhận được có vấn đề?",
        "gold_answer": "Shopee chưa hỗ trợ đổi hàng; người mua có thể từ chối nhận khi đồng kiểm hoặc gửi yêu cầu Trả hàng/Hoàn tiền sau khi nhận hàng.",
        "gold_doc_id": "buyer-return-conditions",
        "answer_markers": ("chưa hỗ trợ yêu cầu đổi hàng", "từ chối nhận hàng khi đồng kiểm"),
        "metadata_filter": {"audience": "buyer"},
    },
    {
        "id": 3,
        "query": "Người mua có thể gửi yêu cầu Trả hàng/Hoàn tiền bằng những cách nào?",
        "gold_answer": "Gửi trực tiếp tại trang đơn hàng hoặc gửi tại mục Trò Chuyện Với Shopee rồi chọn Khiếu nại trả hàng hoàn tiền.",
        "gold_doc_id": "buyer-return-request-guide",
        "answer_markers": ("gửi yêu cầu trực tiếp tại trang đơn hàng", "gửi yêu cầu tại mục trò chuyện với shopee"),
        "metadata_filter": {"audience": "buyer"},
    },
    {
        "id": 4,
        "query": "Sau khi Shopee chấp nhận hoàn tiền, tiền hoàn về thẻ tín dụng hoặc thẻ ghi nợ mất bao lâu?",
        "gold_answer": "Khoảng 7–14 ngày làm việc, tùy theo ngân hàng.",
        "gold_doc_id": "buyer-refund-timeline",
        "answer_markers": ("7 - 14 ngày làm việc", "thẻ tín dụng/ghi nợ"),
        "metadata_filter": {"audience": "buyer"},
    },
    {
        "id": 5,
        "query": "Một yêu cầu hoàn tiền cần được phản hồi trong bao lâu?",
        "gold_answer": "Người Bán có 02 ngày lịch kể từ khi nhận yêu cầu; không phản hồi được xem là đồng ý với quyết định của Shopee.",
        "gold_doc_id": "seller-mall-return-obligations",
        "answer_markers": ("hoàn tiền ngay", "trong vòng 02 ngày lịch"),
        "metadata_filter": {"audience": "seller"},
    },
]


class Chunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...


class HeadingChunker:
    """Split Markdown by headings and recursively split oversized sections."""

    def __init__(self, chunk_size: int = CHUNK_SIZE) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = [
            section.strip()
            for section in re.split(r"(?m)(?=^#{1,6}\s+)", text.strip())
            if section.strip()
        ]
        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue

            first_line, separator, remainder = section.partition("\n")
            heading = first_line.strip() if re.match(r"^#{1,6}\s+", first_line) else ""
            body = remainder.strip() if heading and separator else section
            available_size = max(100, self.chunk_size - len(heading) - 1)
            pieces = RecursiveChunker(chunk_size=available_size).chunk(body)
            for piece in pieces:
                chunks.append(f"{heading}\n{piece}".strip() if heading else piece)
        return chunks


def build_chunker(strategy: str) -> tuple[Chunker, str]:
    if strategy == "fixed":
        return (
            FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
            f"FixedSizeChunker(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})",
        )
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=CHUNK_SIZE), f"RecursiveChunker(chunk_size={CHUNK_SIZE})"
    if strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=3), "SentenceChunker(max_sentences_per_chunk=3)"
    if strategy == "heading":
        return HeadingChunker(chunk_size=CHUNK_SIZE), f"HeadingChunker(chunk_size={CHUNK_SIZE})"
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def parse_scalar(value: str) -> str:
    """Parse the simple quoted or unquoted scalar values used by this corpus."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return str(json.loads(value))
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def load_markdown(path: Path) -> tuple[dict[str, str], str]:
    """Return YAML-like front matter and body content without embedding YAML."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing opening front matter delimiter")

    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"{path}: malformed front matter")

    metadata: dict[str, str] = {}
    for raw_line in parts[1].splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = parse_scalar(value)

    missing = sorted(REQUIRED_METADATA - metadata.keys())
    if missing:
        raise ValueError(f"{path}: missing metadata: {', '.join(missing)}")
    if metadata["doc_id"] != path.stem:
        raise ValueError(f"{path}: doc_id must match file name")

    content = parts[2].strip()
    if not content:
        raise ValueError(f"{path}: empty document body")
    return metadata, content


def build_documents(
    data_dir: Path,
    chunker: Chunker,
    strategy_name: str = "fixed",
) -> tuple[list[Document], dict[str, int]]:
    documents: list[Document] = []
    chunk_counts: dict[str, int] = {}

    for path in sorted(data_dir.glob("*.md")):
        frontmatter, content = load_markdown(path)
        chunks = chunker.chunk(content)
        chunk_counts[path.stem] = len(chunks)

        for index, chunk in enumerate(chunks):
            metadata = {
                **frontmatter,
                "doc_id": path.stem,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "chunking_strategy": strategy_name,
                "source_file": str(path),
            }
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata=metadata,
                )
            )

    if not documents:
        raise ValueError(f"No Markdown documents found in {data_dir}")
    return documents, chunk_counts


def build_embedder(provider: str) -> tuple[Callable[[str], list[float]], str]:
    if provider == "local":
        embedder = LocalEmbedder()
    elif provider == "openai":
        embedder = OpenAIEmbedder()
    elif provider == "gemini":
        embedder = GeminiEmbedder()
    elif provider == "mock":
        embedder = _mock_embed
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    return embedder, getattr(embedder, "_backend_name", embedder.__class__.__name__)


def preview(text: str, limit: int = 260) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def evidence_rank(results: list[dict], markers: tuple[str, ...]) -> int | None:
    """Return the earliest cumulative rank whose context contains all answer markers."""
    context_parts: list[str] = []
    for rank, result in enumerate(results, start=1):
        context_parts.append(result["content"])
        context = " ".join(" ".join(context_parts).lower().split())
        if all(marker.lower() in context for marker in markers):
            return rank
    return None


def print_compact_results(label: str, results: list[dict]) -> None:
    print(label)
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"  {rank}. score={result['score']:.4f} "
            f"doc_id={metadata['doc_id']} chunk={metadata['chunk_index']} "
            f"audience={metadata['audience']}"
        )


def run_benchmark(data_dir: Path, provider: str, strategy: str, top_k: int = 3) -> int:
    chunker, strategy_label = build_chunker(strategy)
    documents, chunk_counts = build_documents(data_dir, chunker, strategy_name=strategy)
    embedder, backend_name = build_embedder(provider)
    store = EmbeddingStore(collection_name="shopee_fixed_size", embedding_fn=embedder)
    store.add_documents(documents)

    print("=== SHOPEE RETRIEVAL BENCHMARK ===")
    print(f"strategy: {strategy_label}")
    print(f"embedding: {backend_name}")
    print(f"source documents: {len(chunk_counts)}")
    print(f"stored chunks: {store.get_collection_size()}")
    print("chunks per document:")
    for doc_id, count in chunk_counts.items():
        print(f"  - {doc_id}: {count}")

    doc_hits = 0
    evidence_hits = 0
    benchmark_score = 0
    for benchmark in BENCHMARKS:
        query = benchmark["query"]
        metadata_filter = benchmark["metadata_filter"]
        results = store.search_with_filter(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        print("\n" + "=" * 80)
        print(f"QUERY {benchmark['id']}: {query}")
        print(f"FILTER: {metadata_filter}")
        print(f"GOLD DOC: {benchmark['gold_doc_id']}")
        print(f"GOLD ANSWER: {benchmark['gold_answer']}")

        gold_rank = None
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            if metadata["doc_id"] == benchmark["gold_doc_id"] and gold_rank is None:
                gold_rank = rank
            print(
                f"{rank}. score={result['score']:.4f} "
                f"doc_id={metadata['doc_id']} "
                f"chunk={metadata['chunk_index']} "
                f"audience={metadata['audience']}"
            )
            print(f"   {preview(result['content'])}")

        if gold_rank is not None:
            doc_hits += 1
            print(f"GOLD DOC IN TOP-{top_k}: YES (rank {gold_rank})")
        else:
            print(f"GOLD DOC IN TOP-{top_k}: NO")

        answer_rank = evidence_rank(results, benchmark["answer_markers"])
        if answer_rank is not None:
            evidence_hits += 1
            points = 2 if answer_rank == 1 else 1
            benchmark_score += points
            print(f"GOLD ANSWER EVIDENCE IN TOP-{top_k}: YES (by rank {answer_rank}, {points} point(s))")
        else:
            print(f"GOLD ANSWER EVIDENCE IN TOP-{top_k}: NO (0 points)")

    print("\n" + "=" * 80)
    filter_benchmark = BENCHMARKS[-1]
    query = filter_benchmark["query"]
    filtered = store.search_with_filter(
        query,
        top_k=top_k,
        metadata_filter=filter_benchmark["metadata_filter"],
    )
    unfiltered = store.search(query, top_k=top_k)
    print("A/B METADATA FILTER — QUERY 5")
    print_compact_results("WITHOUT FILTER:", unfiltered)
    print_compact_results("WITH audience=seller:", filtered)

    print("\n" + "=" * 80)
    print(f"RETRIEVAL DOC HIT RATE: {doc_hits}/{len(BENCHMARKS)}")
    print(f"ANSWER EVIDENCE HIT RATE: {evidence_hits}/{len(BENCHMARKS)}")
    print(f"CONTENT-LEVEL BENCHMARK SCORE: {benchmark_score}/10")
    print("Note: content-level evidence is checked cumulatively over top-k using answer markers.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the personal fixed-size retrieval benchmark.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--provider",
        choices=("local", "mock", "openai", "gemini"),
        default=os.getenv("EMBEDDING_PROVIDER", "local").strip().lower(),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--strategy",
        choices=("fixed", "recursive", "sentence", "heading"),
        default="fixed",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(override=False)
    args = parse_args()
    return run_benchmark(args.data_dir, args.provider, args.strategy, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
