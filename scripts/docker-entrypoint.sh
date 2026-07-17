#!/bin/sh
set -eu

if [ ! -f "/app/data/bioc_chunks.jsonl" ] || [ ! -f "/app/data/bm25_corpus.jsonl" ]; then
  echo "Bootstrapping local indexes from bundled BioC fixture..."
  python -m scripts.bootstrap_indexes --reset
fi

exec "$@"
