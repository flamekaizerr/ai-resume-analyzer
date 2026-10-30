import argparse
import sys

from dotenv import find_dotenv, load_dotenv

from . import qa
from .store import FaissStore


def main(argv=None):
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(prog="python -m rag", description="Ask questions over your resume and JDs")
    parser.add_argument("--index-dir", default=None, help="where the index is stored (default: backend/rag_index)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="index .txt, .md and .pdf files or folders")
    p_ing.add_argument("paths", nargs="+")
    p_ing.add_argument("--chunk-size", type=int, default=500)
    p_ing.add_argument("--overlap", type=int, default=100)

    p_ask = sub.add_parser("ask", help="ask a question against the index")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=4, help="number of chunks to retrieve")

    args = parser.parse_args(argv)
    folder = args.index_dir or qa.index_dir()

    from .embeddings import get_embedder

    embedder = get_embedder()

    if args.cmd == "ingest":
        store = qa.ingest(args.paths, embedder, folder=folder, chunk_size=args.chunk_size, overlap=args.overlap)
        files = sorted({c.source for c in store.chunks})
        print(f"Indexed {len(store)} chunks from {len(files)} files into {folder}")
        for f in files:
            print(f"  {f}")
        return 0

    try:
        store = FaissStore.load(folder)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    result = qa.answer(args.question, store, embedder, k=args.k)
    warning = qa.jd_warning(args.question, {chunk.source for chunk in store.chunks})
    if warning:
        print(f"Warning: {warning}")
    print(result["answer"])
    if result["mode"] == "model":
        print("\nSources:")
        for s in result["sources"]:
            print(f"  {s['source']} #{s['chunk_id']} (score {s['score']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
