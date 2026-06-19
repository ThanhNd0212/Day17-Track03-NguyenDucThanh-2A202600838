from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


class FactEntry(NamedTuple):
    """A single extracted fact with metadata for confidence filtering."""
    value: str
    confidence: float   # 0.0–1.0
    is_correction: bool  # True when the message explicitly corrects old info


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    # Heuristic: ~4 characters per token is a common approximation for English/Vietnamese
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def __post_init__(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        # TODO: slugify or sanitize the user id before building the file path.
        # Replace characters unsafe for filenames with underscores
        safe_id = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        # TODO: return file content or an empty default markdown profile.
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# Profile: {user_id}\n\n(No information recorded yet)\n"

    def write_text(self, user_id: str, content: str) -> Path:
        # TODO: write markdown to disk and return the file path.
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        # TODO: replace one occurrence inside User.md and return whether it changed.
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        # TODO: return the current file size in bytes.
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Return a dict of key->value pairs parsed from User.md bullet lines."""
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for line in text.splitlines():
            # Match lines like `- **key**: value` or `- key: value`
            m = re.match(r"-\s+\*{0,2}(.+?)\*{0,2}:\s*(.+)", line.strip())
            if m:
                result[m.group(1).strip().lower()] = m.group(2).strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str, is_correction: bool = False) -> None:
        """Insert or update a fact line in User.md.

        Conflict handling: if a fact already exists and `is_correction` is True,
        the old value is overwritten and the update is annotated so reviewers can
        audit how the profile evolved over time.
        """
        text = self.read_text(user_id)
        pattern = re.compile(
            r"(- \*{0,2}" + re.escape(key) + r"\*{0,2}: ?)(.+)", re.IGNORECASE
        )

        if is_correction:
            # Annotate the new line to signal an explicit user correction
            new_line = f"- **{key}**: {value}  _(đã đính chính)_"
        else:
            new_line = f"- **{key}**: {value}"

        if pattern.search(text):
            # Conflict handling: overwrite existing fact.
            # Correction always wins; normal upsert overwrites silently.
            text = pattern.sub(new_line, text, count=1)
        else:
            if text.endswith("\n"):
                text = text + new_line + "\n"
            else:
                text = text + "\n" + new_line + "\n"
        self.write_text(user_id, text)


def extract_profile_updates_scored(message: str) -> dict[str, FactEntry]:
    """Internal function that extracts facts with confidence scores and correction flags.

    Confidence levels used:
    - 0.95  correction phrase detected ("đính chính", "không còn X", "giờ là Y")
    - 0.80  explicit direct declaration ("tên là X", "làm Y engineer")
    - 0.60  indirect or implied mention (keyword appears in context)

    Only entries returned by this function are eligible to be written to User.md.
    Callers use `extract_profile_updates()` which applies the threshold filter.
    """

    # Skip turns that are purely questions — they don't contain facts to store
    stripped = message.strip()
    question_only = bool(re.fullmatch(r"[^.!]*\?", stripped))
    if question_only:
        return {}

    # Detect correction context once for the whole message.
    # Phrases like "đính chính", "không còn X nữa", "giờ là Y" signal corrections.
    _CORRECTION_SIGNALS = re.compile(
        r"đính\s+chính|không\s+còn\s+\w+\s+nữa|giờ\s+(?:là|chuyển|mình)|"
        r"thực\s+ra|cập\s+nhật.*thông\s+tin|thay\s+đổi",
        re.IGNORECASE,
    )
    is_correction_context = bool(_CORRECTION_SIGNALS.search(message))
    # Confidence boost: corrections are written with highest priority
    correction_conf = 0.95 if is_correction_context else 0.0

    scored: dict[str, FactEntry] = {}

    # --- Name (confidence 0.80 — requires explicit "tên" phrase) ---
    _NAME_STOPWORDS = {
        "là", "tên", "mình", "tôi", "bạn", "đang", "không", "có", "và",
        "backend", "engineer", "python", "ai", "cà", "phê", "ở", "tại",
        "hay", "được", "thì", "những", "các", "rất", "cũng", "vẫn",
    }
    name_patterns = [
        r"tên\s+(?:mình|tôi)\s+là\s+([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_]{1,30})(?:[,.\s]|$)",
        r"tên\s+là\s+([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_]{1,30})(?:[,.\s]|$)",
        r"(?:mình|tôi)\s+tên\s+(?:là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_]{1,30})(?:[,.\s]|$)",
        r"chào\b.*?tên\s+(?:là\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_]{1,30})(?:[,.\s]|$)",
    ]
    for pat in name_patterns:
        m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
        if m:
            candidate = m.group(1).strip().rstrip(".,")
            if candidate.lower() not in _NAME_STOPWORDS and len(candidate) >= 2:
                scored["name"] = FactEntry(candidate, 0.80, is_correction_context)
            break

    # --- Location (correction → 0.95, direct statement → 0.80) ---
    # Vietnamese city names can be multi-word (Đà Nẵng, Hồ Chí Minh, Hà Nội).
    # Capture group allows internal spaces; terminators are conjunctions / punctuation.
    _LOC_CAP = r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{0,25}?)"
    _LOC_END = r"(?:\s+(?:chứ|và|nhé|để|vì|mà|nên|trong|vài|nhưng|để|theo)|[,.]|$)"
    _LOCATION_STOPWORDS = {"đâu", "đó", "này", "kia", "nhà", "công", "ty", "đây"}
    # Correction patterns checked first — they carry is_correction=True
    correction_loc_patterns = [
        rf"(?:giờ|hiện\s+tại|thực\s+ra|đính\s+chính)\b.*?(?:mình|tôi)\s+(?:đang\s+)?(?:ở|tại|sống\s+ở)\s+{_LOC_CAP}{_LOC_END}",
        rf"(?:mình|tôi)\s+(?:đang\s+)?(?:ở|tại)\s+{_LOC_CAP}\s+(?:trong|vài|chứ)\b",
    ]
    direct_loc_patterns = [
        rf"(?:mình|tôi)\s+(?:đang\s+)?(?:ở|tại)\s+{_LOC_CAP}{_LOC_END}",
        rf"(?:mình|tôi)\s+sống\s+ở\s+{_LOC_CAP}{_LOC_END}",
        rf"(?:nơi\s+ở|địa\s+chỉ)\s*(?:là|:)?\s*{_LOC_CAP}{_LOC_END}",
        rf"hiện\s+(?:đang\s+)?(?:ở|tại)\s+{_LOC_CAP}{_LOC_END}",
    ]
    loc_found = False
    for pat in correction_loc_patterns:
        m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
        if m:
            candidate = m.group(1).strip().rstrip(".,")
            if len(candidate) >= 2 and candidate.lower() not in _LOCATION_STOPWORDS:
                scored["location"] = FactEntry(candidate, correction_conf or 0.90, True)
                loc_found = True
            break
    if not loc_found:
        for pat in direct_loc_patterns:
            m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
            if m:
                candidate = m.group(1).strip().rstrip(".,")
                if len(candidate) >= 2 and candidate.lower() not in _LOCATION_STOPWORDS:
                    scored["location"] = FactEntry(candidate, 0.80, False)
                break

    # --- Profession (confidence 0.80; correction context → 0.95) ---
    job_titles = re.findall(
        r"\b(MLOps\s+engineer|backend\s+engineer|frontend\s+engineer|full[\-\s]?stack\s+engineer|"
        r"data\s+scientist|product\s+manager|software\s+engineer|ML\s+engineer|"
        r"DevOps\s+engineer|AI\s+engineer)\b",
        message, re.IGNORECASE,
    )
    if job_titles:
        # Take LAST title — handles "không còn là backend, giờ là MLOps engineer"
        title = job_titles[-1]
        conf = correction_conf if correction_conf else 0.80
        scored["profession"] = FactEntry(title, conf, is_correction_context)

    # --- Favorite drink (confidence 0.80 — keyword is very specific) ---
    if "cà phê sữa đá" in message.lower():
        scored["favorite_drink"] = FactEntry("cà phê sữa đá", 0.80, False)
    else:
        m = re.search(
            r"(?:đồ\s+uống|thích\s+uống|hay\s+uống)\s+(?:yêu\s+thích\s+(?:là\s+)?)?(cà\s+phê[^\n,.]*)(?:[,.\n]|$)",
            message, re.IGNORECASE,
        )
        if m:
            scored["favorite_drink"] = FactEntry(m.group(1).strip().rstrip(".,"), 0.80, False)

    # --- Favorite food (confidence 0.80) ---
    if "mì quảng" in message.lower():
        scored["favorite_food"] = FactEntry("mì Quảng", 0.80, False)
    else:
        food_m = re.search(
            r"(?:món\s+ăn\s+yêu\s+thích|món\s+ruột)\s+(?:là\s+)?([\w\sÀ-ỹ]{3,30})(?:[,.\n]|$)",
            message, re.IGNORECASE,
        )
        if food_m:
            scored["favorite_food"] = FactEntry(food_m.group(1).strip().rstrip(".,"), 0.80, False)

    # --- Pet (confidence 0.80 — very specific pattern) ---
    pet_m = re.search(r"(?:nuôi|có)\s+(?:một?\s+)?(?:bé\s+)?(\w+)\s+tên\s+(\w+)", message, re.IGNORECASE)
    if pet_m:
        scored["pet"] = FactEntry(f"{pet_m.group(1)} tên {pet_m.group(2)}", 0.80, False)

    # --- Response style (confidence 0.75 — style keywords are reliable signals) ---
    style_keywords = ["ngắn gọn", "3 bullet", "bullet", "ví dụ thực tế", "ví dụ thực chiến", "trade-off", "rõ ý", "có cấu trúc"]
    found_styles = [kw for kw in style_keywords if kw in message.lower()]
    if found_styles:
        scored["response_style"] = FactEntry(", ".join(found_styles), 0.75, False)

    # --- Interests (confidence 0.60 — keyword may appear in passing context) ---
    interests = []
    for kw in ["Python", "AI", "MLOps", "RAG", "LangChain", "benchmark", "memory"]:
        if kw.lower() in message.lower():
            interests.append(kw)
    if interests:
        scored["interests"] = FactEntry(", ".join(interests), 0.60, False)

    return scored


def extract_profile_updates(message: str, confidence_threshold: float = 0.70) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.

    BONUS — Confidence threshold:
    Only facts whose confidence score >= `confidence_threshold` are returned.
    Default threshold = 0.70 means indirect keyword mentions (score 0.60) are
    silently dropped, preventing low-quality facts from polluting User.md.
    Set threshold = 0.0 to disable filtering (returns everything found).
    """

    scored = extract_profile_updates_scored(message)
    return {
        key: entry.value
        for key, entry in scored.items()
        if entry.confidence >= confidence_threshold
    }


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""

    # Take a representative sample from the messages to summarize
    sample = messages[:max_items]
    lines = []
    for msg in sample:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        # Keep first 100 chars of each message for the summary
        short = content[:100].replace("\n", " ")
        if len(content) > 100:
            short += "..."
        lines.append(f"[{role}] {short}")

    return "Tóm tắt hội thoại cũ:\n" + "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        self._init_thread(thread_id)
        self.state[thread_id]["messages"].append({"role": role, "content": content})
        self._maybe_compact(thread_id)

    # Maximum characters the rolling summary is allowed to grow to.
    # Keeps the summary bounded so prompt cost stays sub-linear on long threads.
    _MAX_SUMMARY_CHARS: int = 600

    def _maybe_compact(self, thread_id: str) -> None:
        """Compact old messages into a summary when total tokens exceed threshold."""
        thread = self.state[thread_id]
        messages: list[dict] = thread["messages"]

        # Estimate total tokens across all messages + existing summary
        total = estimate_tokens(thread["summary"])
        for msg in messages:
            total += estimate_tokens(msg["content"])

        if total <= self.threshold_tokens:
            return

        # Messages to keep verbatim (most recent ones)
        keep_n = self.keep_messages
        if len(messages) <= keep_n:
            return

        to_summarize = messages[:-keep_n]
        to_keep = messages[-keep_n:]

        # Build new summary from old summary + messages being compacted.
        # Cap at _MAX_SUMMARY_CHARS to keep prompt cost sub-linear on long threads.
        old_summary = thread["summary"]
        new_portion = summarize_messages(to_summarize)
        combined = (old_summary + "\n\n" + new_portion) if old_summary else new_portion
        if len(combined) > self._MAX_SUMMARY_CHARS:
            # Keep the tail (most recent) so context is the most relevant
            combined = "...(lược bỏ phần cũ)...\n" + combined[-self._MAX_SUMMARY_CHARS:]
        thread["summary"] = combined

        thread["messages"] = to_keep
        thread["compactions"] = thread["compactions"] + 1

    def context(self, thread_id: str) -> dict[str, object]:
        # TODO: return per-thread state with keys like messages, summary, compactions.
        self._init_thread(thread_id)
        return dict(self.state[thread_id])

    def compaction_count(self, thread_id: str) -> int:
        # TODO: return number of compactions for this thread.
        self._init_thread(thread_id)
        return self.state[thread_id]["compactions"]
