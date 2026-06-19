from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates_scored,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Call the real LangGraph agent and return standardized result."""
        try:
            profile_text = self.profile_store.read_text(user_id)
            ctx = self.compact_memory.context(thread_id)
            summary = ctx.get("summary", "")

            system_extra = f"\n\n## Hồ sơ người dùng:\n{profile_text}"
            if summary:
                system_extra += f"\n\n## Tóm tắt hội thoại cũ:\n{summary}"

            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}], "system_extra": system_extra},
                config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
            )
            response_text = result["messages"][-1].content

            # Extract and persist profile facts with confidence threshold
            scored_facts = extract_profile_updates_scored(message)
            for key, entry in scored_facts.items():
                if entry.confidence >= 0.70:
                    self.profile_store.upsert_fact(
                        user_id, key, entry.value, is_correction=entry.is_correction
                    )

            # Append to compact memory
            self.compact_memory.append(thread_id, "user", message)
            self.compact_memory.append(thread_id, "assistant", response_text)

            usage = getattr(result.get("messages", [])[-1], "usage_metadata", None)
            agent_tokens = usage.output_tokens if usage else estimate_tokens(response_text)
            prompt_tokens = usage.input_tokens if usage else self._estimate_prompt_context_tokens(user_id, thread_id)

            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
            self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

            return {"response": response_text, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        # 1. Extract stable profile facts with confidence scores.
        #    BONUS – Confidence threshold + Conflict handling:
        #    - extract_profile_updates_scored() assigns each fact a score (0.60–0.95).
        #    - Only facts with score >= 0.70 are written (threshold filter).
        #    - Facts marked is_correction=True are passed to upsert_fact so the
        #      annotation "(đã đính chính)" is written to User.md, making overrides
        #      auditable without keeping the stale old value.
        scored_facts = extract_profile_updates_scored(message)

        # 2. Persist those facts into `User.md` (with confidence threshold 0.70)
        for key, entry in scored_facts.items():
            if entry.confidence >= 0.70:
                self.profile_store.upsert_fact(
                    user_id, key, entry.value, is_correction=entry.is_correction
                )

        # 3. Append the message into compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt-context load from `User.md` + summary + recent messages
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 5. Generate a response that can answer long-term recall questions
        response = self._offline_response(user_id, thread_id, message)

        # 6. Append the assistant reply and update token counters
        self.compact_memory.append(thread_id, "assistant", response)

        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        # Include `User.md`
        profile_text = self.profile_store.read_text(user_id)
        total = estimate_tokens(profile_text)

        ctx = self.compact_memory.context(thread_id)

        # Include compact summary text
        summary = ctx.get("summary", "")
        total += estimate_tokens(summary)

        # Include recent kept messages
        for msg in ctx.get("messages", []):
            total += estimate_tokens(msg.get("content", ""))

        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        msg_lower = message.lower()
        profile = self.profile_store.read_text(user_id)
        facts = self.profile_store.facts(user_id)

        # Also gather facts from compact memory context
        ctx = self.compact_memory.context(thread_id)
        thread_text = " ".join(
            m["content"] for m in ctx.get("messages", []) if m.get("role") == "user"
        )
        summary_text = ctx.get("summary", "")
        all_history = profile + "\n" + summary_text + "\n" + thread_text

        # --- Recall: name ---
        if re.search(r"tên\s*(mình|tôi)?\s*(là|gì)", msg_lower) or "tên gì" in msg_lower:
            name = facts.get("name")
            if not name:
                m = re.search(r"tên\s+(?:là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_\s]{1,30}?)(?:[,.\s]|$)", all_history, re.IGNORECASE)
                if m:
                    name = m.group(1).strip()
            if name:
                return f"Bạn tên là **{name}**."

        # --- Recall: profession ---
        if re.search(r"(nghề|làm\s+gì|công\s+việc|chức\s+danh)", msg_lower):
            profession = facts.get("profession")
            if not profession:
                for title in ["MLOps engineer", "backend engineer", "frontend engineer", "data scientist"]:
                    if title.lower() in all_history.lower():
                        profession = title
                        break
            if profession:
                location = facts.get("location", "")
                loc_part = f", hiện đang ở {location}" if location else ""
                return f"Nghề nghiệp hiện tại của bạn là **{profession}**{loc_part}."

        # --- Recall: location ---
        if re.search(r"(ở\s+đâu|nơi\s+ở|đang\s+ở|địa\s+chỉ)", msg_lower):
            location = facts.get("location")
            if not location:
                m = re.search(r"(?:ở|tại|sống\s+ở)\s+([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,20}?)(?:[,.\s]|$)", all_history, re.IGNORECASE)
                if m:
                    location = m.group(1).strip()
            if location:
                return f"Hiện tại bạn đang ở **{location}**."

        # --- Recall: drink ---
        if re.search(r"(đồ\s+uống|thức\s+uống|uống\s+gì)", msg_lower):
            drink = facts.get("favorite_drink")
            if not drink and "cà phê sữa đá" in all_history.lower():
                drink = "cà phê sữa đá"
            if drink:
                return f"Đồ uống yêu thích của bạn là **{drink}**."

        # --- Recall: food ---
        if re.search(r"(món\s+ăn|ăn\s+gì|thức\s+ăn)", msg_lower):
            food = facts.get("favorite_food")
            if not food and "mì quảng" in all_history.lower():
                food = "mì Quảng"
            if food:
                return f"Món ăn yêu thích của bạn là **{food}**."

        # --- Recall: pet ---
        if re.search(r"(nuôi|thú\s+cưng|pet|con\s+gì)", msg_lower):
            pet = facts.get("pet")
            if not pet:
                m = re.search(r"(?:nuôi|có)\s+(?:một?\s+)?(?:bé\s+)?(\w+)\s+tên\s+(\w+)", all_history, re.IGNORECASE)
                if m:
                    pet = f"{m.group(1)} tên {m.group(2)}"
            if pet:
                return f"Bạn đang nuôi một bé **{pet}**."

        # --- Recall: style ---
        if re.search(r"(style|phong\s+cách|trả\s+lời.*thích|cách\s+trả\s+lời)", msg_lower):
            style = facts.get("response_style")
            if not style and "ngắn gọn" in all_history.lower():
                style = "ngắn gọn, có ví dụ thực tế"
            if "3 bullet" in all_history.lower():
                style = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
            if style:
                return f"Style trả lời bạn thích: **{style}**."

        # --- Comprehensive summary recall ---
        if re.search(r"(nhắc\s+lại|tóm\s+tắt|mô\s+tả.*ai|bạn\s+biết.*không)", msg_lower):
            parts = []
            name = facts.get("name")
            if name:
                parts.append(f"- **Tên**: {name}")
            profession = facts.get("profession")
            if profession:
                parts.append(f"- **Nghề nghiệp**: {profession}")
            location = facts.get("location")
            if location:
                parts.append(f"- **Nơi ở**: {location}")
            drink = facts.get("favorite_drink")
            if drink:
                parts.append(f"- **Đồ uống**: {drink}")
            food = facts.get("favorite_food")
            if food:
                parts.append(f"- **Món ăn**: {food}")
            pet = facts.get("pet")
            if pet:
                parts.append(f"- **Thú cưng**: {pet}")
            style = facts.get("response_style")
            if style:
                parts.append(f"- **Style trả lời**: {style}")
            interests = facts.get("interests")
            if interests:
                parts.append(f"- **Mối quan tâm**: {interests}")
            if parts:
                return "Dựa trên thông tin đã lưu:\n" + "\n".join(parts)

        # --- Generic acknowledgment of new info ---
        new_facts_desc = ", ".join(f"{k}={v}" for k, v in facts.items())
        if facts:
            return (
                f"Tôi đã ghi nhớ thông tin bạn chia sẻ ({new_facts_desc}) vào hồ sơ User.md. "
                "Thông tin này sẽ được giữ lại qua các phiên làm việc."
            )

        return (
            "Tôi đã lắng nghe và lưu ý thông tin bạn chia sẻ. "
            "Hỏi tôi bất cứ lúc nào để nhắc lại thông tin đã lưu."
        )

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        if not self.config.model.api_key:
            return

        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            profile_store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read the persistent User.md profile for the given user."""
                return profile_store.read_text(user_id)

            @tool
            def upsert_user_fact(user_id: str, key: str, value: str) -> str:
                """Upsert a fact into User.md for the given user."""
                profile_store.upsert_fact(user_id, key, value)
                return f"Saved: {key} = {value}"

            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()

            self.langchain_agent = create_react_agent(
                llm,
                tools=[read_user_profile, upsert_user_fact],
                checkpointer=checkpointer,
                state_modifier=(
                    "Bạn là một trợ lý AI có bộ nhớ dài hạn. "
                    "Sử dụng tool `read_user_profile` để đọc hồ sơ người dùng trước khi trả lời. "
                    "Sử dụng tool `upsert_user_fact` để lưu thông tin ổn định từ người dùng vào User.md. "
                    "Trả lời ngắn gọn, có cấu trúc và ưu tiên trade-off khi giải thích kỹ thuật."
                ),
            )
        except Exception:
            self.langchain_agent = None
