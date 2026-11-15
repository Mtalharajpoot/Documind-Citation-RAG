import csv
import json
import os
import time

from rag_pipeline import RAGPipeline, FALLBACK

EXPERIMENTS = [("exp1", "config1", 3), ("exp2", "config2", 5)]
OUT = "outputs/evaluation_results"


def run():
    os.makedirs(OUT, exist_ok=True)
    with open("data/evaluation_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    summary = {}
    for name, config, k in EXPERIMENTS:
        print(f"\n=== {name}: {config}, top_k={k} ===")
        rag = RAGPipeline(config, top_k=k)
        rows = []
        for q in questions:
            try:
                out = rag.ask(q["question"])
            except Exception as e:
                out = {"answer": f"ERROR: {e}", "sources": [], "chunks": []}

            ans = out["answer"]
            refused = ans.strip() == FALLBACK
            src_docs = {s["document"] for s in out["sources"]}
            retr_docs = {c["document"] for c in out["chunks"]}
            sources = "; ".join(f"{s['document']} p{s['page']}" for s in out["sources"])

            if q["answerable"]:
                kw_ok = (not refused) and all(kw.lower() in ans.lower() for kw in q["keywords"])
                cite_ok = (q["document"] in src_docs) if not refused else False
                retr_ok = q["document"] in retr_docs
                refusal_ok = ""
            else:
                kw_ok, cite_ok, retr_ok = "", "", ""
                refusal_ok = refused

            rows.append({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "expected_answer": q["expected_answer"], "generated_answer": ans,
                "sources": sources, "auto_correct": kw_ok, "citation_doc_ok": cite_ok,
                "retrieval_doc_ok": retr_ok, "correct_refusal": refusal_ok,
                "manual_check": "",
            })
            print(f"  Q{q['id']:>2} done")
            time.sleep(6)

        with open(f"{OUT}/results_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        ans_rows = [r for r in rows if r["auto_correct"] != ""]
        unans_rows = [r for r in rows if r["correct_refusal"] != ""]
        correct_ans = sum(1 for r in ans_rows if r["auto_correct"] is True)
        correct_ref = sum(1 for r in unans_rows if r["correct_refusal"] is True)
        checked = [r for r in ans_rows if r["generated_answer"].strip() != FALLBACK]
        correct_cit = sum(1 for r in checked if r["citation_doc_ok"] is True)
        retr_hit = sum(1 for r in ans_rows if r["retrieval_doc_ok"] is True)

        summary[name] = {
            "config": config, "top_k": k,
            "answerable_correct": f"{correct_ans}/{len(ans_rows)}",
            "correct_refusals": f"{correct_ref}/{len(unans_rows)}",
            "answer_correctness_rate": round((correct_ans + correct_ref) / len(rows), 3),
            "citation_correctness_rate": round(correct_cit / len(checked), 3) if checked else 0,
            "unanswerable_handling_rate": round(correct_ref / len(unans_rows), 3),
            "retrieval_doc_hit_rate": round(retr_hit / len(ans_rows), 3),
        }
        print(summary[name])

    with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved in", OUT)


if __name__ == "__main__":
    run()