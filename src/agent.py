from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        context_parts = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = (
                metadata.get("source_url")
                or metadata.get("source")
                or metadata.get("doc_id")
                or result.get("id", "unknown")
            )
            context_parts.append(
                f"[{index}] Nguồn: {source}\n{result.get('content', '')}"
            )
        context = "\n\n".join(context_parts)
        prompt = (
            "Bạn là trợ lý hỏi đáp dựa trên tài liệu được cung cấp.\n"
            "Chỉ sử dụng thông tin trong CONTEXT để trả lời câu hỏi. "
            "Không suy đoán hoặc bịa thêm thông tin. "
            "Nếu CONTEXT không đủ thông tin, hãy nói rõ không tìm thấy câu trả lời. "
            "Trích dẫn số nguồn tương ứng, ví dụ [1] hoặc [2].\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"CÂU HỎI: {question}\n"
            "TRẢ LỜI:"
        )
        return self.llm_fn(prompt)
