#!/usr/bin/env bash
# katzbot\rebuild_katzbot.sh
set -euo pipefail

# Run this from your project root:
# D:/Yeshiva/Spring26/intership/ideathon/code/paperMind

echo "Cleaning old KatzBot caches..."
rm -f katzbot/events_cache.json
rm -rf katzbot/faiss_index
rm -f katzbot/pages_cache_all.json
rm -f katzbot/pages_cache_katz.json
rm -f katzbot/crawl_failures_all.json
rm -f katzbot/crawl_failures_katz.json

echo "Rebuilding KatzBot index..."
python katzbot/build_index.py --refresh --test


# chmod +x rebuild_katzbot.sh
# ./katzbot/rebuild_katzbot.sh