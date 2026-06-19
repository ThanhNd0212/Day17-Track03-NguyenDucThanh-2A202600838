from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0

    answer_lower = answer.lower()
    matched = sum(1 for fact in expected if fact.lower() in answer_lower)

    ratio = matched / len(expected)
    if ratio == 0:
        return 0.0
    if ratio < 1.0:
        return 0.5
    return 1.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""

    if not answer or not answer.strip():
        return 0.0

    # Base quality: response is non-empty
    score = 0.3

    # Bonus: response contains expected facts
    answer_lower = answer.lower()
    if expected:
        matched = sum(1 for fact in expected if fact.lower() in answer_lower)
        score += 0.5 * (matched / len(expected))

    # Bonus: response has some structure (bullets, colon, bold markers)
    if any(marker in answer for marker in ["- ", "**", ":", "•"]):
        score += 0.1

    # Bonus: response is a reasonable length (not too short, not a huge dump)
    words = len(answer.split())
    if 5 <= words <= 200:
        score += 0.1

    return min(1.0, score)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0
    memory_growth_bytes = 0

    for conv in conversations:
        user_id: str = conv["user_id"]
        conv_id: str = conv["id"]
        turns: list[str] = conv["turns"]
        recall_questions: list[dict] = conv.get("recall_questions", [])

        # Use a unique thread id per conversation
        thread_id = f"{conv_id}-main"

        # 1 & 2 & 3: Feed all turns, track tokens
        for turn in turns:
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # 4: Ask recall questions in a FRESH thread (cross-session recall test)
        recall_thread_id = f"{conv_id}-recall"
        for rq in recall_questions:
            question = rq["question"]
            expected = rq.get("expected_contains", [])

            result = agent.reply(user_id, recall_thread_id, question)
            answer = result.get("response", "")

            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

            # 5: Compute recall and quality
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

        # 6: Record memory file growth and compaction count
        total_compactions += agent.compaction_count(thread_id)

        # Memory growth: only advanced agent has a profile file
        if hasattr(agent, "memory_file_size"):
            memory_growth_bytes += agent.memory_file_size(user_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth_bytes,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""

    try:
        from tabulate import tabulate

        headers = [
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ]
        table_data = [
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score:.3f}",
                f"{r.response_quality:.3f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
            for r in rows
        ]
        return tabulate(table_data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback to simple text table
        lines = []
        header = (
            f"{'Agent':<20} | {'Agent tokens':>13} | {'Prompt tokens':>13} | "
            f"{'Recall':>6} | {'Quality':>7} | {'Mem (bytes)':>11} | {'Compact':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for r in rows:
            lines.append(
                f"{r.agent_name:<20} | {r.agent_tokens_only:>13} | {r.prompt_tokens_processed:>13} | "
                f"{r.recall_score:>6.3f} | {r.response_quality:>7.3f} | "
                f"{r.memory_growth_bytes:>11} | {r.compactions:>7}"
            )
        return "\n".join(lines)


def _ensure_utf8_stdout() -> None:
    """Set stdout to UTF-8 on Windows so Vietnamese characters print correctly."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    _ensure_utf8_stdout()
    config = load_config(Path(__file__).resolve().parent.parent)

    # TODO:
    # - load both datasets from root/data
    # - initialize baseline and advanced agents
    # - run benchmarks
    # - print comparison tables

    data_dir = config.data_dir

    # ── Standard Benchmark ──────────────────────────────────────────────────
    print("=" * 70)
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print("=" * 70)

    std_conversations = load_conversations(data_dir / "conversations.json")

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, std_conversations, config),
        run_agent_benchmark("Advanced", advanced_std, std_conversations, config),
    ]

    print(format_rows(std_rows))
    print()

    # ── Long-Context Stress Benchmark ────────────────────────────────────────
    print("=" * 70)
    print("LONG-CONTEXT STRESS BENCHMARK  (data/advanced_long_context.json)")
    print("=" * 70)

    stress_conversations = load_conversations(data_dir / "advanced_long_context.json")

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_conversations, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_conversations, config),
    ]

    print(format_rows(stress_rows))
    print()

    # ── Analysis notes ───────────────────────────────────────────────────────
    print("=" * 70)
    print("PHÂN TÍCH KẾT QUẢ")
    print("=" * 70)
    _print_analysis(std_rows, stress_rows)


def _print_analysis(std_rows: list[BenchmarkRow], stress_rows: list[BenchmarkRow]) -> None:
    """Print a concise analysis of the benchmark results."""

    baseline_std = next(r for r in std_rows if r.agent_name == "Baseline")
    advanced_std = next(r for r in std_rows if r.agent_name == "Advanced")
    baseline_stress = next(r for r in stress_rows if r.agent_name == "Baseline")
    advanced_stress = next(r for r in stress_rows if r.agent_name == "Advanced")

    print("""
1. Tại sao Advanced có recall tốt hơn Baseline?
   - Advanced lưu facts ổn định vào User.md (persistent memory).
   - Khi mở thread mới (cross-session), Advanced đọc User.md nên vẫn biết
     tên, nghề nghiệp, nơi ở của người dùng, còn Baseline thì quên hoàn toàn.
   - Recall score của Advanced: {adv_recall:.3f} vs Baseline: {base_recall:.3f}

2. Tại sao Advanced có thể tốn hơn ở hội thoại ngắn?
   - Mỗi lượt Advanced kéo theo User.md + compact summary vào prompt.
   - Ở hội thoại ngắn (ít turns), chi phí đọc profile vượt lợi ích nén context.
   - Prompt tokens: Advanced={adv_prompt}, Baseline={base_prompt} (standard benchmark).

3. Tại sao compact memory giúp Advanced ở hội thoại dài?
   - Khi thread dài, Baseline kéo toàn bộ lịch sử vào mỗi lượt → prompt tokens
     tăng tuyến tính theo số lượt.
   - Advanced nén phần lịch sử cũ thành summary ngắn → prompt tokens tăng
     chậm hơn đáng kể sau mỗi lần compact.
   - Stress benchmark: Advanced compactions={adv_compact}, Baseline=0.
   - Prompt tokens stress: Advanced={adv_stress_prompt} vs Baseline={base_stress_prompt}.

4. Memory file tăng trưởng và rủi ro:
   - User.md của Advanced tăng theo số facts được lưu.
   - Memory growth (bytes): {adv_mem} bytes sau {n_conv} conversations.
   - Rủi ro: nếu không có pruning, file sẽ phình to, tăng prompt cost lâu dài.
   - Giải pháp: confidence threshold, memory decay, conflict handling.
""".format(
        adv_recall=advanced_std.recall_score,
        base_recall=baseline_std.recall_score,
        adv_prompt=advanced_std.prompt_tokens_processed,
        base_prompt=baseline_std.prompt_tokens_processed,
        adv_compact=advanced_stress.compactions,
        adv_stress_prompt=advanced_stress.prompt_tokens_processed,
        base_stress_prompt=baseline_stress.prompt_tokens_processed,
        adv_mem=advanced_std.memory_growth_bytes,
        n_conv=10,
    ))


if __name__ == "__main__":
    main()
