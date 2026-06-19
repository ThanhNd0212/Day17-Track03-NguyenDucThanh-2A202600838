from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates_scored


def make_config(tmp_path: Path) -> LabConfig:
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests

    base_config = load_config()
    return LabConfig(
        base_dir=base_config.base_dir,
        data_dir=base_config.data_dir,
        # Isolate state so tests don't touch real profiles
        state_dir=tmp_path / "state",
        # Low threshold so compaction triggers with just a few messages
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=base_config.model,
        judge_model=base_config.judge_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"

    # --- Read: empty profile returns default content ---
    default = store.read_text(user_id)
    assert "test_user" in default or "No information" in default, \
        "Default profile should mention the user id or be empty placeholder"

    # --- Write: explicit content round-trips correctly ---
    content = "# Profile: test_user\n\n- **name**: DũngCT\n- **location**: Huế\n"
    path = store.write_text(user_id, content)
    assert path.exists(), "write_text should create the file on disk"
    assert store.read_text(user_id) == content, "read_text should return what was written"

    # --- Edit: replace one occurrence and verify ---
    changed = store.edit_text(user_id, "Huế", "Đà Nẵng")
    assert changed, "edit_text should return True when the string is found and replaced"
    updated = store.read_text(user_id)
    assert "Đà Nẵng" in updated, "Updated location should appear in file"
    assert "Huế" not in updated, "Old location should be gone after edit"

    # --- Edit: no-op when search string not present ---
    not_changed = store.edit_text(user_id, "Huế", "Hà Nội")
    assert not not_changed, "edit_text should return False when the string is not found"

    # --- file_size: reflects bytes on disk ---
    size = store.file_size(user_id)
    assert size > 0, "file_size should be positive after writing content"

    print("PASS test_user_markdown_read_write_edit")


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    # Low threshold = compaction fires after just a few messages
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "thread-compact-test"

    # Fill the thread with enough messages to exceed the threshold
    long_messages = [
        "Mình tên là DũngCT và đang làm MLOps engineer tại Huế.",
        "Mình thích Python, AI ứng dụng và cà phê sữa đá.",
        "Tuần này mình đang ôn lại async Python để cải thiện pipeline.",
        "Mình muốn câu trả lời ngắn gọn, rõ ý và có ví dụ thực tế.",
        "Cuối tuần mình hay đi biển Mỹ Khê chụp ảnh phong cảnh.",
        "Mục tiêu quý này là xây agent nhớ người dùng tốt hơn.",
    ]

    for msg in long_messages:
        manager.append(thread_id, "user", msg)
        manager.append(thread_id, "assistant", "Đã ghi nhận thông tin.")

    compactions = manager.compaction_count(thread_id)
    assert compactions >= 1, (
        f"Expected at least 1 compaction with threshold=50 tokens, got {compactions}. "
        "Check that _maybe_compact fires when total tokens exceed the threshold."
    )

    ctx = manager.context(thread_id)
    # After compaction, only keep_messages (2) should remain verbatim
    assert len(ctx["messages"]) <= 2, (
        f"After compaction, should keep at most 2 messages, got {len(ctx['messages'])}"
    )
    # Summary should be non-empty
    assert ctx["summary"], "Summary should be set after compaction"

    print(f"PASS test_compact_trigger  (compactions={compactions})")


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    user_id = "dungct"

    # ── Session 1: teach the advanced agent some facts ──────────────────────
    advanced = AdvancedAgent(config=config, force_offline=True)
    advanced.reply(user_id, "session-1", "Mình tên là DũngCT.")
    advanced.reply(user_id, "session-1", "Mình đang làm MLOps engineer ở Huế.")
    advanced.reply(user_id, "session-1", "Đồ uống yêu thích là cà phê sữa đá.")

    # ── Session 2 (new thread, same user): advanced should still recall ──────
    adv_result = advanced.reply(user_id, "session-2", "Mình tên gì và đang ở đâu?")
    adv_response = adv_result["response"].lower()

    assert "dungct" in adv_response or "dũngct" in adv_response, (
        f"Advanced agent should recall name 'DũngCT' in new session. Got: {adv_result['response']!r}"
    )

    # ── Baseline: teach same facts in session-1 ──────────────────────────────
    baseline = BaselineAgent(config=config, force_offline=True)
    baseline.reply(user_id, "session-1", "Mình tên là DũngCT.")
    baseline.reply(user_id, "session-1", "Mình đang làm MLOps engineer ở Huế.")

    # ── Baseline session-2: should NOT know the name ─────────────────────────
    base_result = baseline.reply(user_id, "session-2", "Mình tên gì?")
    base_response = base_result["response"].lower()

    # Baseline should not recall facts from a different thread
    assert "dungct" not in base_response and "dũngct" not in base_response, (
        f"Baseline should NOT recall name across sessions. Got: {base_result['response']!r}"
    )

    print("PASS test_cross_session_recall")


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    user_id = "stress_user"
    thread_id = "long-thread"

    # Build a longer message sequence to stress the context window
    messages = [
        f"Đây là tin nhắn số {i}: mình đang chia sẻ nhiều thông tin để test compact memory. "
        f"Mình tên DũngCT, làm MLOps, ở Huế, thích Python và AI ứng dụng."
        for i in range(1, 15)
    ]

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    for msg in messages:
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    base_prompt = baseline.prompt_token_usage(thread_id)
    adv_prompt = advanced.prompt_token_usage(thread_id)
    adv_compactions = advanced.compaction_count(thread_id)

    # Advanced should have triggered at least one compaction
    assert adv_compactions >= 1, (
        f"Expected at least 1 compaction on a long thread, got {adv_compactions}. "
        "Check CompactMemoryManager threshold in make_config."
    )

    # Advanced's prompt cost should be less than baseline's because of compaction
    # (baseline grows linearly, advanced compresses old turns)
    assert adv_prompt < base_prompt, (
        f"Advanced prompt tokens ({adv_prompt}) should be less than Baseline ({base_prompt}) "
        "after compaction on a long thread."
    )

    print(
        f"PASS test_compact_reduces_prompt_load_on_long_thread  "
        f"(baseline={base_prompt} tokens, advanced={adv_prompt} tokens, "
        f"compactions={adv_compactions})"
    )


def test_confidence_threshold_filters_low_quality_facts() -> None:
    """BONUS: verify that low-confidence facts (score < 0.70) are not written to User.md.

    'Interests' are scored 0.60 — below the default threshold of 0.70.
    A message that only mentions "Python" in passing should NOT add an interests
    entry to User.md, while an explicit name declaration (score 0.80) should pass.
    """

    # Message with an explicit name (score 0.80 → passes) and an interest keyword
    # that appears purely in passing context (score 0.60 → filtered out).
    message = "Tên mình là DũngCT. Hôm nay mình đọc tài liệu về Python."

    scored = extract_profile_updates_scored(message)

    # Verify scoring: name should be high-confidence, interests should be low
    assert "name" in scored, "Name should be extracted from explicit 'tên là' phrase"
    assert scored["name"].confidence >= 0.70, "Name confidence should be >= 0.70"

    assert "interests" in scored, "Interests (Python) should appear in scored output"
    assert scored["interests"].confidence < 0.70, (
        "Interests confidence should be < 0.70 (indirect mention only)"
    )

    # extract_profile_updates with default threshold 0.70 should keep name, drop interests
    from memory_store import extract_profile_updates
    filtered = extract_profile_updates(message, confidence_threshold=0.70)
    assert "name" in filtered, "Name should survive the 0.70 threshold"
    assert "interests" not in filtered, (
        "Interests should be filtered out at threshold=0.70 "
        "(prevents vague keyword mentions from polluting User.md)"
    )

    # With threshold 0.0, both should appear
    all_facts = extract_profile_updates(message, confidence_threshold=0.0)
    assert "interests" in all_facts, "With threshold=0.0, all facts should be returned"

    print("PASS test_confidence_threshold_filters_low_quality_facts")


def test_conflict_handling_correction_overrides(tmp_path: Path) -> None:
    """BONUS: verify that explicit corrections overwrite old facts in User.md.

    When a user says 'đính chính: giờ mình ở Đà Nẵng', the new location should
    replace the old one (Huế), and the entry should be annotated as a correction.
    The stale value must NOT remain in the file alongside the new one.
    """

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "conflict_test_user"

    # ── Step 1: initial fact ─────────────────────────────────────────────────
    store.upsert_fact(user_id, "location", "Huế")
    assert "Huế" in store.read_text(user_id), "Initial location should be Huế"

    # ── Step 2: correction message ───────────────────────────────────────────
    correction_msg = (
        "Mình đính chính một chút: thực ra giờ mình đang ở Đà Nẵng "
        "chứ không còn ở Huế nữa."
    )
    scored = extract_profile_updates_scored(correction_msg)

    assert "location" in scored, "Location should be extracted from the correction message"
    assert scored["location"].is_correction, (
        "is_correction should be True when correction phrases are detected"
    )
    assert scored["location"].confidence >= 0.70, (
        "Correction should have high confidence (>= 0.70)"
    )

    # Persist with is_correction flag
    entry = scored["location"]
    store.upsert_fact(user_id, "location", entry.value, is_correction=entry.is_correction)

    profile_text = store.read_text(user_id)

    # New value must be present
    assert "Đà Nẵng" in profile_text, "New location Đà Nẵng should be in User.md"

    # Old value must be gone — conflict resolution, not accumulation
    assert "Huế" not in profile_text, (
        "Old location Huế should be overwritten, not kept alongside new value"
    )

    # Correction annotation must be present
    assert "đính chính" in profile_text.lower(), (
        "Correction should be annotated in User.md for auditability"
    )

    print("PASS test_conflict_handling_correction_overrides")


# ── Run all tests manually when executed as a script ────────────────────────
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_user_markdown_read_write_edit(tmp_path / "t1")
        test_compact_trigger(tmp_path / "t2")
        test_cross_session_recall(tmp_path / "t3")
        test_compact_reduces_prompt_load_on_long_thread(tmp_path / "t4")
        test_confidence_threshold_filters_low_quality_facts()
        test_conflict_handling_correction_overrides(tmp_path / "t5")

    print("\nAll tests passed.")
