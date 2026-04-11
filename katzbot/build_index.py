"""
katzbot/build_index.py
=======================
Standalone CLI script to build the KatzBot FAISS index.

Usage:
    python katzbot/build_index.py              # build (uses cache)
    python katzbot/build_index.py --refresh    # force re-crawl and rebuild
    python katzbot/build_index.py --test       # build + run test queries
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def build(force_refresh: bool = False) -> dict:
    print("\n" + "═" * 60)
    print("  ScholarMind — KatzBot Index Builder")
    print("  Based on website_bot_2_rag_37.ipynb (upgraded 2026)")
    print("═" * 60 + "\n")

    t0 = time.time()
    from katzbot.rag_engine import get_engine

    engine = get_engine()
    stats = engine.build(force_refresh=force_refresh)
    elapsed = time.time() - t0

    print(f"\n✅ Build complete in {elapsed:.1f}s")
    print(f"   Web pages:     {stats['web_pages']}")
    print(f"   Faculty docs:  {stats['faculty_docs']}")
    print(f"   Event docs:    {stats.get('events_docs', 0)}")
    print(f"   Total docs:    {stats['total_docs']}")
    print(f"   FAISS vectors: {stats['index_vectors']:,}")
    print(f"   Index saved:   {stats['index_dir']}")
    print(f"   Docs .pkl:     {stats['pkl_path']}")
    print(f"   Meta  .pkl:    {stats.get('meta_path', 'N/A')}")
    print("   Runtime load:  FAISS disk cache (fast startup)")
    return stats


def run_tests(engine) -> None:
    test_qs = [
        "Who is Prof. Honggang Wang?",
        "What programs does Katz School offer?",
        "What is the curriculum in M.S. in Artificial Intelligence?",
        "How do I apply to the PhD in Computer Science?",
        "What is tuition at Katz School?",
        "What percentage of students get financial aid?",
        "Who should I contact about cybersecurity research?",
        "What events are coming up at Katz School?",
    ]

    print("\n" + "═" * 60)
    print("  Test Queries")
    print("═" * 60)

    for i, q in enumerate(test_qs, 1):
        print(f"\n[{i}] Q: {q}")
        result = engine.ask(q)
        print(f"     A: {result['answer'][:250]}…")
        if result["sources"]:
            print(f"     Sources: {result['sources'][0]}")
        if result["faculty_matches"]:
            f = result["faculty_matches"][0]
            print(f"     Faculty: {f['name']} ({f['email']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force re-crawl and rebuild")
    parser.add_argument("--test", action="store_true", help="Run test queries after building")
    args = parser.parse_args()

    build(force_refresh=args.refresh)
    if args.test:
        from katzbot.rag_engine import get_engine
        run_tests(get_engine())
