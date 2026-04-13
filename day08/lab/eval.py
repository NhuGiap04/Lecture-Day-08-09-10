"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
  Đổi đồng thời chunking + hybrid + rerank + prompt = không biết biến nào có tác dụng.
"""

import json
import csv
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
JUDGE_MAX_CONTEXT_CHARS = int(os.getenv("JUDGE_MAX_CONTEXT_CHARS", "5000"))

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3 — điều chỉnh theo lựa chọn của nhóm)
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "query_transform_strategy": "expansion",
    "label": "variant_hybrid_rerank_qexpand",
}


# =============================================================================
# SCORING FUNCTIONS
# 4 metrics từ slide: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _contains_abstain(answer: str) -> bool:
    lowered = (answer or "").lower()
    markers = [
        "không đủ dữ liệu",
        "khong du du lieu",
        "không biết",
        "khong biet",
        "insufficient",
        "i don't know",
    ]
    return any(m in lowered for m in markers)


def _safe_score_1_to_5(value: float) -> int:
    return max(1, min(5, int(round(value))))


def _call_llm_judge(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Gọi LLM-as-a-judge và kỳ vọng output JSON:
      {"score": <1-5>, "notes": "..."}
    """
    if not USE_LLM_JUDGE:
        return None

    openai_key = os.getenv("OPENAI_API_KEY")
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_base_url = os.getenv("NVIDIA_BASE_URL")

    # if not openai_key:
    #     return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=nvidia_api_key, base_url=nvidia_base_url)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge for RAG outputs. "
                        "Return only JSON with keys: score (int 1-5), notes (string)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        # Cố parse trực tiếp; nếu có code block thì bóc JSON trong block.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            parsed = json.loads(match.group(0)) if match else None

        if not isinstance(parsed, dict):
            return None

        score = parsed.get("score")
        notes = parsed.get("notes", "")
        if score is None:
            return None

        return {
            "score": _safe_score_1_to_5(float(score)),
            "notes": str(notes),
        }
    except Exception:
        return None


def _judge_faithfulness_llm(answer: str, chunks_used: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not chunks_used:
        return None

    context_lines = []
    for i, c in enumerate(chunks_used[:5], 1):
        meta = c.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        text = (c.get("text", "") or "")[:1200]
        context_lines.append(f"[{i}] {source} | {section}\n{text}")

    context = "\n\n".join(context_lines)
    context = context[:JUDGE_MAX_CONTEXT_CHARS]

    prompt = (
        "Metric: Faithfulness (1-5).\n"
        "Definition: score 5 if answer is fully grounded in provided context, "
        "score 1 if mostly unsupported/hallucinated.\n"
        f"Answer:\n{answer}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Return JSON only."
    )
    return _call_llm_judge(prompt)


def _judge_relevance_llm(query: str, answer: str) -> Optional[Dict[str, Any]]:
    prompt = (
        "Metric: Answer Relevance (1-5).\n"
        "Definition: score 5 if answer directly and correctly addresses the user question; "
        "score 1 if irrelevant.\n"
        f"Question:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        "Return JSON only."
    )
    return _call_llm_judge(prompt)


def _judge_completeness_llm(query: str, answer: str, expected_answer: str) -> Optional[Dict[str, Any]]:
    prompt = (
        "Metric: Completeness (1-5).\n"
        "Definition: compare model answer vs expected answer; score 5 if key points are covered, "
        "score 1 if most key points are missing.\n"
        f"Question:\n{query}\n\n"
        f"Expected Answer:\n{expected_answer}\n\n"
        f"Model Answer:\n{answer}\n\n"
        "Return JSON only."
    )
    return _call_llm_judge(prompt)

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness: Câu trả lời có bám đúng chứng cứ đã retrieve không?
    Câu hỏi: Model có tự bịa thêm thông tin ngoài retrieved context không?

    Thang điểm 1-5:
      5: Mọi thông tin trong answer đều có trong retrieved chunks
      4: Gần như hoàn toàn grounded, 1 chi tiết nhỏ chưa chắc chắn
      3: Phần lớn grounded, một số thông tin có thể từ model knowledge
      2: Nhiều thông tin không có trong retrieved chunks
      1: Câu trả lời không grounded, phần lớn là model bịa

    Có thể chấm theo 2 cách:

    Cách 1 — Chấm thủ công (Manual, đơn giản):
        Đọc answer và chunks_used, chấm điểm theo thang trên.
        Ghi lý do ngắn gọn vào "notes".

    Cách 2 — LLM-as-Judge (Tự động, nâng cao):
        Gửi prompt cho LLM:
            "Given these retrieved chunks: {chunks}
             And this answer: {answer}
             Rate the faithfulness on a scale of 1-5.
             5 = completely grounded in the provided context.
             1 = answer contains information not in the context.
             Output JSON: {'score': <int>, 'reason': '<string>'}"

    Trả về dict với: score (1-5) và notes (lý do)
    """
    if not answer or answer.startswith("ERROR:"):
        return {
            "score": 1,
            "notes": "Không có câu trả lời hợp lệ để đánh giá faithfulness",
        }

    if _contains_abstain(answer):
        return {
            "score": 5,
            "notes": "Model đã abstain, giảm nguy cơ hallucination",
        }

    if not chunks_used:
        return {
            "score": 1,
            "notes": "Không có chunk nào được dùng",
        }

    judged = _judge_faithfulness_llm(answer, chunks_used)
    if judged:
        print("Using LLM-as-a-Judge")
        judged["notes"] = f"[LLM-Judge:{JUDGE_MODEL}] {judged.get('notes', '')}"
        return judged

    print("Using heuristic evaluation")
    answer_tokens = set(_tokenize(answer))
    context_text = "\n".join(c.get("text", "") for c in chunks_used)
    context_tokens = set(_tokenize(context_text))

    if not answer_tokens:
        return {
            "score": 1,
            "notes": "Answer rỗng sau tokenize",
        }

    overlap_ratio = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    score = _safe_score_1_to_5(1 + overlap_ratio * 4)

    return {
        "score": score,
        "notes": f"Token overlap answer/context = {overlap_ratio:.2f}",
    }


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance: Answer có trả lời đúng câu hỏi người dùng hỏi không?
    Câu hỏi: Model có bị lạc đề hay trả lời đúng vấn đề cốt lõi không?

    Thang điểm 1-5:
      5: Answer trả lời trực tiếp và đầy đủ câu hỏi
      4: Trả lời đúng nhưng thiếu vài chi tiết phụ
      3: Trả lời có liên quan nhưng chưa đúng trọng tâm
      2: Trả lời lạc đề một phần
      1: Không trả lời câu hỏi

    """
    if not answer or answer.startswith("ERROR:"):
        return {
            "score": 1,
            "notes": "Không có câu trả lời hợp lệ",
        }

    if _contains_abstain(answer):
        # Abstain vẫn có thể phù hợp nếu query thiếu context, chấm trung tính-khá
        return {
            "score": 4,
            "notes": "Answer dạng abstain",
        }

    judged = _judge_relevance_llm(query, answer)
    if judged:
        judged["notes"] = f"[LLM-Judge:{JUDGE_MODEL}] {judged.get('notes', '')}"
        return judged

    q_tokens = set(_tokenize(query))
    a_tokens = set(_tokenize(answer))
    if not q_tokens:
        return {
            "score": 3,
            "notes": "Query không đủ token để đánh giá",
        }

    overlap_ratio = len(q_tokens & a_tokens) / len(q_tokens)
    score = _safe_score_1_to_5(1 + overlap_ratio * 4)

    return {
        "score": score,
        "notes": f"Query-answer token overlap = {overlap_ratio:.2f}",
    }


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Câu hỏi: Expected source có nằm trong retrieved chunks không?

    Đây là metric đo retrieval quality, không phải generation quality.

    Cách tính đơn giản:
        recall = (số expected source được retrieve) / (tổng số expected sources)

    Ví dụ:
        expected_sources = ["policy/refund-v4.pdf", "sla-p1-2026.pdf"]
        retrieved_sources = ["policy/refund-v4.pdf", "helpdesk-faq.md"]
        recall = 1/2 = 0.5

    Metric này đã được implement theo expected_sources matching.
    """
    if not expected_sources:
        # Câu hỏi không có expected source (ví dụ: "Không đủ dữ liệu" cases)
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    # Matching theo partial path để chịu được khác biệt format source path.
    found = 0
    missing = []
    for expected in expected_sources:
        # Kiểm tra partial match (tên file)
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    return {
        "score": round(recall * 5),  # Convert to 1-5 scale
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness: Answer có thiếu điều kiện ngoại lệ hoặc bước quan trọng không?
    Câu hỏi: Answer có bao phủ đủ thông tin so với expected_answer không?

    Thang điểm 1-5:
      5: Answer bao gồm đủ tất cả điểm quan trọng trong expected_answer
      4: Thiếu 1 chi tiết nhỏ
      3: Thiếu một số thông tin quan trọng
      2: Thiếu nhiều thông tin quan trọng
      1: Thiếu phần lớn nội dung cốt lõi

    """
    if not answer or answer.startswith("ERROR:"):
        return {
            "score": 1,
            "notes": "Không có câu trả lời hợp lệ",
        }

    if not expected_answer:
        return {
            "score": 3,
            "notes": "Không có expected_answer để đối chiếu",
        }

    if _contains_abstain(answer):
        # Nếu expected có nguồn rỗng (khó trả lời), abstain là hợp lý hơn.
        return {
            "score": 4,
            "notes": "Answer abstain; cần review thủ công cho case thiếu context",
        }

    judged = _judge_completeness_llm(query, answer, expected_answer)
    if judged:
        judged["notes"] = f"[LLM-Judge:{JUDGE_MODEL}] {judged.get('notes', '')}"
        return judged

    expected_tokens = set(_tokenize(expected_answer))
    answer_tokens = set(_tokenize(answer))

    if not expected_tokens:
        return {
            "score": 3,
            "notes": "Expected answer không đủ token",
        }

    coverage = len(expected_tokens & answer_tokens) / len(expected_tokens)
    score = _safe_score_1_to_5(1 + coverage * 4)

    missing = sorted(list(expected_tokens - answer_tokens))[:8]

    return {
        "score": score,
        "notes": f"Expected coverage={coverage:.2f}; missing sample={missing}",
    }


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm.

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row

     Hàm đã implement đầy đủ load → run pipeline → scoring → aggregate.
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                query_transform_strategy=config.get("query_transform_strategy"),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:100]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"\nAverage {metric}: {avg:.2f}" if avg else f"\nAverage {metric}: N/A (chưa chấm)")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.

    Hàm in bảng so sánh baseline/variant theo metric và theo từng câu.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg and v_avg) else None

        b_str = f"{b_avg:.2f}" if b_avg else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg else "N/A"
        d_str = f"{delta:+.2f}" if delta else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([
            str(b_row.get(m, "?")) for m in metrics
        ])
        v_scores_str = "/".join([
            str(v_row.get(m, "?")) for m in metrics
        ])

        # So sánh đơn giản
        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """
    Tạo báo cáo tóm tắt scorecard dạng markdown.

    Template markdown cho báo cáo scorecard.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')[:50]} |\n")

    return md


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # Kiểm tra test questions
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi")

        # In preview
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")

    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline ---")
    print("Lưu ý: Cần hoàn thành Sprint 2 trước khi chạy scorecard!")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )

        # Save scorecard
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
        scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
        scorecard_path.write_text(baseline_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {scorecard_path}")

    except NotImplementedError:
        print("Pipeline chưa implement. Hoàn thành Sprint 2 trước.")
        baseline_results = []

    # --- Chạy Variant ---
    print("\n--- Chạy Variant ---")
    try:
        variant_results = run_scorecard(
            config=VARIANT_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
        variant_path = RESULTS_DIR / "scorecard_variant.md"
        variant_path.write_text(variant_md, encoding="utf-8")
        print(f"\nScorecard variant lưu tại: {variant_path}")
    except Exception as e:
        print(f"Lỗi khi chạy variant: {e}")
        variant_results = []

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv"
        )
    else:
        print("\nBỏ qua A/B comparison vì chưa có đủ baseline và variant results.")

    print("\n\nViệc cần làm Sprint 4:")
    print("  1. Hoàn thành Sprint 2 + 3 trước")
    print("  2. Chấm điểm thủ công hoặc implement LLM-as-Judge trong score_* functions")
    print("  3. Chạy run_scorecard(BASELINE_CONFIG)")
    print("  4. Chạy run_scorecard(VARIANT_CONFIG)")
    print("  5. Gọi compare_ab() để thấy delta")
    print("  6. Cập nhật docs/tuning-log.md với kết quả và nhận xét")
