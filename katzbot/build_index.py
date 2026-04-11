"""
katzbot/build_index.py
=======================
Standalone script to build the KatzBot index.
Equivalent to running all notebook cells in sequence.

Usage:
    python katzbot/build_index.py              # build (uses cache if fresh)
    python katzbot/build_index.py --refresh    # force re-crawl
    python katzbot/build_index.py --test       # build + run test queries

This is the production version of the notebook's step-by-step execution.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def build(force_refresh: bool = False) -> dict:
    """Full index build pipeline — equivalent to running the notebook end-to-end."""
    print("\n" + "=" * 60)
    print("  KatzBot Index Builder — PaperMind 2026")
    print("  (Evolved from website_bot_2_rag_37.ipynb)")
    print("=" * 60 + "\n")

    t0 = time.time()

    from katzbot.rag_engine import get_engine
    engine = get_engine()
    stats  = engine.build(force_refresh=force_refresh)

    elapsed = time.time() - t0
    print(f"\n✅ Build complete in {elapsed:.1f}s")
    print(f"   Web pages crawled:  {stats['web_pages']}")
    print(f"   Faculty documents:  {stats['faculty_docs']}")
    print(f"   Total documents:    {stats['total_docs']}")
    print(f"   FAISS vectors:      {stats['index_vectors']}")
    print(f"   Index saved to:     {stats['index_dir']}")
    return stats


def run_test_queries(engine) -> None:
    """
    Run the same test questions used in the original notebook evaluation.
    Mirrors the notebook's 'Compare the results' section.
    """
    test_questions = [
        "Who is Honggang Wang?",
        "What programs does Katz School offer?",
        "What is the curriculum in M.S. in Artificial Intelligence?",
        "How can I apply to the PhD in computer science?",
        "How much is tuition at Katz School?",
        "What is the student/faculty ratio at this university?",
        "What percentage of students get financial aid?",
        "Who should I contact about cybersecurity research?",
    ]

    print("\n" + "=" * 60)
    print("  Test Queries (mirrors notebook evaluation)")
    print("=" * 60)

    for i, q in enumerate(test_questions, 1):
        print(f"\n[{i}] Q: {q}")
        result = engine.ask(q)
        print(f"     A: {result['answer'][:300]}...")
        if result["sources"]:
            print(f"     Sources: {', '.join(result['sources'][:2])}")
        if result["faculty_matches"]:
            fac = result["faculty_matches"][0]
            print(f"     Faculty: {fac['name']} ({fac['email']})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KatzBot FAISS index")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-crawl (ignore cache)")
    parser.add_argument("--test", action="store_true",
                        help="Run test queries after building")
    args = parser.parse_args()

    stats = build(force_refresh=args.refresh)

    if args.test:
        from katzbot.rag_engine import get_engine
        engine = get_engine()
        run_test_queries(engine)