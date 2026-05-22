"""
Conversation history manager with automatic token-aware compression.

How compression works
---------------------
After `max_turns` Q/A exchanges, the oldest turns are summarised into a
single SystemMessage. Only the last 2 Q/A pairs are kept verbatim.

Without compression a 10-turn conversation grows to ~10 × (Q + A) ≈ 2 000+
tokens of history on every request. With compression the oldest portion is
collapsed to a ~150-token summary, keeping history overhead roughly constant.
"""

from typing import List, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

AnyMessage = Union[HumanMessage, AIMessage, SystemMessage]

_VERBATIM_PAIRS = 2  # Keep the last N Q/A pairs as raw messages


class HistoryManager:
    """
    Thread-local conversation memory with automatic compression.

    Usage::

        history = HistoryManager(model, max_turns=6)
        history.add(user_question, assistant_answer)
        messages = history.messages  # pass to model.invoke()
        history.clear()
    """

    def __init__(self, model: ChatOpenAI, max_turns: int = 6) -> None:
        self._model = model
        self._max_turns = max_turns
        self._history: List[AnyMessage] = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def messages(self) -> List[AnyMessage]:
        return list(self._history)

    @property
    def is_empty(self) -> bool:
        return len(self._history) == 0

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._history if isinstance(m, HumanMessage))

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, human: str, ai: str) -> None:
        """Append a Q/A pair and compress if the turn limit is exceeded."""
        self._history.append(HumanMessage(content=human))
        self._history.append(AIMessage(content=ai))
        if self.turn_count > self._max_turns:
            self._compress()

    def clear(self) -> None:
        self._history = []

    # ── Compression ───────────────────────────────────────────────────────────

    def _compress(self) -> None:
        """
        Summarise all but the last `_VERBATIM_PAIRS` Q/A pairs.
        The summary is stored as a SystemMessage at the front of history.
        """
        keep = _VERBATIM_PAIRS * 2  # number of messages to keep verbatim
        to_compress = self._history[:-keep]
        recent = self._history[-keep:]

        lines = []
        for msg in to_compress:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                lines.append(f"[Context]: {msg.content}")

        prompt = (
            "Summarise the following conversation concisely. "
            "Preserve key technical facts, spec values, and conclusions. "
            "Be brief (≤ 100 words).\n\n"
            + "\n".join(lines)
        )

        try:
            summary = self._model.invoke([
                SystemMessage(content="You are a concise conversation summariser."),
                HumanMessage(content=prompt),
            ]).content
        except Exception:
            summary = "[prior conversation — details unavailable]"

        self._history = [
            SystemMessage(content=f"[Earlier conversation summary]: {summary}")
        ] + recent
