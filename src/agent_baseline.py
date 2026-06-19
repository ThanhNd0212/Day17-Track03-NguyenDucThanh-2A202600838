from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Call the real LangGraph agent and return standardized result."""
        try:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            response_text = result["messages"][-1].content
            usage = getattr(result.get("messages", [])[-1], "usage_metadata", None)
            agent_tokens = usage.output_tokens if usage else estimate_tokens(response_text)
            prompt_tokens = usage.input_tokens if usage else 0

            sess = self._get_session(thread_id)
            sess.token_usage += agent_tokens
            sess.prompt_tokens_processed += prompt_tokens

            return {"response": response_text, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}
        except Exception:
            return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self._get_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self._get_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        sess = self._get_session(thread_id)

        # Append user message to this thread's history
        sess.messages.append({"role": "user", "content": message})

        # Estimate prompt context: system prompt stub + all messages in this thread
        system_stub = "Bạn là một trợ lý AI hữu ích. Bạn chỉ nhớ thông tin trong cuộc trò chuyện hiện tại."
        prompt_tokens = estimate_tokens(system_stub)
        for msg in sess.messages:
            prompt_tokens += estimate_tokens(msg["content"])

        # Generate a simple deterministic reply based on the message content
        response = self._generate_response(sess.messages, message)

        # Append assistant reply
        sess.messages.append({"role": "assistant", "content": response})

        agent_tokens = estimate_tokens(response)
        sess.token_usage += agent_tokens
        # Baseline keeps entire thread history in each prompt — prompt tokens grow linearly
        sess.prompt_tokens_processed += prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _generate_response(self, history: list[dict], message: str) -> str:
        """Generate a deterministic reply from within-session context only."""
        msg_lower = message.lower()

        # Search current session for facts mentioned earlier in this thread
        session_text = " ".join(m["content"] for m in history[:-1])  # exclude current message

        # Try to answer recall questions from session history
        if "tên" in msg_lower and ("gì" in msg_lower or "là" in msg_lower):
            import re
            m = re.search(r"tên\s+(?:là\s+|mình\s+là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_\s]{1,30}?)(?:[,.\s]|$)", session_text, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                return f"Trong cuộc trò chuyện này, bạn đã giới thiệu tên là {name}."

        if "đồ uống" in msg_lower or "uống" in msg_lower:
            if "cà phê sữa đá" in session_text.lower():
                return "Đồ uống yêu thích bạn đã nhắc trong cuộc trò chuyện này là cà phê sữa đá."

        if "ở đâu" in msg_lower or "nơi ở" in msg_lower or "đang ở" in msg_lower:
            import re
            m = re.search(r"(?:ở|tại|sống\s+ở)\s+([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,20}?)(?:[,.\s]|$)", session_text, re.IGNORECASE)
            if m:
                loc = m.group(1).strip()
                return f"Trong cuộc trò chuyện này, bạn đề cập đang ở {loc}."

        if "nghề" in msg_lower or "làm gì" in msg_lower:
            for title in ["MLOps engineer", "backend engineer", "software engineer"]:
                if title.lower() in session_text.lower():
                    return f"Trong cuộc trò chuyện này, bạn cho biết đang làm {title}."

        # Generic acknowledgment
        if len(history) <= 1:
            return "Xin chào! Tôi sẵn sàng hỗ trợ bạn. Lưu ý tôi chỉ nhớ thông tin trong cuộc trò chuyện này."

        return (
            "Tôi đã ghi nhận thông tin bạn chia sẻ trong cuộc trò chuyện này. "
            "Tuy nhiên, tôi không lưu thông tin qua các phiên làm việc khác nhau."
        )

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        # Only attempt to build if we have an API key configured
        if not self.config.model.api_key:
            return

        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()

            # Baseline: no tools, no persistent memory — pure within-session recall
            self.langchain_agent = create_react_agent(
                llm,
                tools=[],
                checkpointer=checkpointer,
                state_modifier=(
                    "Bạn là một trợ lý AI hữu ích. "
                    "Bạn chỉ nhớ thông tin trong cuộc trò chuyện hiện tại. "
                    "Đừng giả vờ nhớ thông tin từ các phiên trước."
                ),
            )
        except Exception:
            self.langchain_agent = None
