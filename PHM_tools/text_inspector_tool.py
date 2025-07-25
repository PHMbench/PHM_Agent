from __future__ import annotations

"""Tool to read local files as text for LLM consumption."""

from pathlib import Path
from typing import Any

from smolagents import Tool
from smolagents.models import Model, MessageRole

try:
    from bs4 import BeautifulSoup
except Exception as exc:  # pragma: no cover - optional import
    BeautifulSoup = None

try:
    import pdfminer.high_level
except Exception:  # pragma: no cover - optional import
    pdfminer = None


class TextInspectorTool(Tool):
    """Read a document and optionally answer a question about it."""

    name = "inspect_file_as_text"
    description = (
        "Load a file and return its textual content. If a question is provided, "
        "summarise the file and answer the question."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read."
        },
        "question": {
            "type": "string",
            "description": "Optional question about the file.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model | None = None, text_limit: int = 100000) -> None:
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".html", ".htm"} and BeautifulSoup is not None:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator=" ")
        if suffix == ".pdf" and pdfminer is not None:
            with open(path, "rb") as f:
                return pdfminer.high_level.extract_text(f) or ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ------------------------------------------------------------------
    def forward(self, file_path: str, question: str | None = None) -> str:
        path = Path(file_path)
        text = self._read_text(path)
        if not question or self.model is None:
            return text[: self.text_limit]
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the file content:\n" + text[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {"type": "text", "text": question}
                ],
            },
        ]
        return self.model(messages).content


if __name__ == "__main__":
    tool = TextInspectorTool()
    content = tool.forward(__file__)
    print("First 60 chars:\n", content[:60])


__all__ = ["TextInspectorTool"]
