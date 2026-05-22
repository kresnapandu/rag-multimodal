"""
AI-enhanced multimodal summary generator (GPT-4o Vision).

Called during ingestion when a document chunk contains tables or images.
Produces a single searchable English text block that captures:
  - Technical spec values from tables
  - Visual information from diagrams / block diagrams / graphs
  - Telecom-domain keywords for improved retrieval
"""

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_SYSTEM = (
    "You create highly searchable English descriptions for telecom hardware documents. "
    "Prioritize technical precision and retrieval coverage."
)

_TASK = (
    "Generate a comprehensive searchable description covering:\n"
    "1) Technical specs (frequency bands, power levels, data rates, "
    "interfaces, VSWR, impedance, antenna gain)\n"
    "2) Model / part numbers and product type\n"
    "3) Supported standards (5G NR, LTE, WCDMA, CPRI, eCPRI, OTN, etc.)\n"
    "4) Visual content analysis (block diagrams, charts, spec tables)\n"
    "5) Technical questions this content can answer\n"
    "Output in English only."
)


def create_multimodal_summary(
    model: ChatOpenAI,
    text: str,
    tables: List[str],
    images_b64: List[str],
) -> str:
    """
    Generate a dense, searchable summary from mixed text + tables + images.

    The summary is stored as an enriched chunk in ChromaDB alongside the
    original semantic text chunks, giving retrieval two shots at finding
    the right content (raw text OR enriched summary).

    Args:
        model: A ChatOpenAI instance with a vision-capable model (gpt-4o).
        text: Raw plain text from the document section.
        tables: List of HTML table strings extracted by Unstructured.
        images_b64: List of base64 JPEG strings extracted by Unstructured.

    Returns:
        Searchable English description string.
    """
    prompt = f"TEXT:\n{text or '(none)'}\n"
    if tables:
        prompt += "\nTABLES:\n"
        for i, tbl in enumerate(tables, 1):
            prompt += f"\nTable {i}:\n{tbl}\n"
    prompt += f"\nTASK:\n{_TASK}\n\nSEARCHABLE DESCRIPTION:"

    content: list = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=content)]
    response = model.invoke(messages)
    return (response.content or "").strip()
