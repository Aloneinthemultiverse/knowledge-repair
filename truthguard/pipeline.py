"""FR-1 — ingestion pipeline: corpus dir -> provenance-tagged chunks + report.

PDF   -> ocr.extract_pdf_pages (native / tier-1 OCR / tier-2 escalation)
MD/TXT-> read directly (page=1, extraction=native)
DOCX  -> markitdown if available, else python-docx, else skipped (reported)

Dedupe (edge case A4): exact SHA1 on normalized text.
Output: storage/truthguard/chunks.json + ingest_report printed and returned.

Run:  python -m truthguard.pipeline [corpus_dir]
"""
import os
import sys
import json
import re
import hashlib

from . import config
from . import ocr
from . import chunker


# Office and web formats parsed by Unstructured-IO. PDF deliberately stays on
# the existing pdfplumber + OCR-ladder path, which already handles scans and
# reports per-page confidence — unstructured's PDF strategies would either lose
# that provenance or pull in heavy layout models.
_UNSTRUCTURED_EXT = {
    "docx": "docx", "doc": "docx", "pptx": "pptx", "xlsx": "xlsx",
    "html": "html", "htm": "html", "epub": "epub", "odt": "odt",
}


def _read_unstructured(path: str, ext: str) -> str:
    """Parse with Unstructured-IO, keeping its element types as Markdown
    structure. Titles become headings and tables keep their HTML, so the
    chunker sees document structure instead of one flat blob of text."""
    mod = _UNSTRUCTURED_EXT[ext]
    import importlib
    partition = getattr(
        importlib.import_module(f"unstructured.partition.{mod}"), f"partition_{mod}")
    out = []
    for el in partition(filename=path):
        kind, text = type(el).__name__, str(el).strip()
        if not text:
            continue
        if kind == "Title":
            out.append(f"\n## {text}\n")
        elif kind == "ListItem":
            out.append(f"- {text}")
        elif kind == "Table":
            # keep the HTML when unstructured produced it; it survives chunking
            # far better than a table flattened into prose
            html = getattr(getattr(el, "metadata", None), "text_as_html", None)
            out.append(html or text)
        else:
            out.append(text)
    return "\n".join(out).strip() or None


def _read_docx(path: str) -> str:
    """Unstructured-IO first, then markitdown, then python-docx.

    Each fallback loses structure, so they are ordered best-first rather than
    lightest-first; a missing optional dependency should degrade the output,
    not the ingest.
    """
    try:
        return _read_unstructured(path, "docx")
    except Exception:
        pass
    try:
        from markitdown import MarkItDown
        return MarkItDown().convert(path).text_content
    except Exception:
        pass
    try:
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    except Exception:
        return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def ingest_corpus(corpus_dir: str = None, storage_dir: str = None,
                  namespace: str = None):
    """namespace tags every chunk so retrieval can scope a tenant's documents.
    Defaults to the environment, so single-tenant corpora are unaffected."""
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    corpus_dir = corpus_dir or config.CORPUS_DIR
    storage_dir = storage_dir or config.STORAGE_DIR
    os.makedirs(storage_dir, exist_ok=True)

    chunks, report = [], {"files": [], "pages": {"native": 0, "ocr": 0, "failed": 0},
                          "tier2_pages": 0, "skipped": [], "deduped": 0}
    seen_hashes = set()

    for fname in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, fname)
        if not os.path.isfile(path):
            continue
        ext = fname.lower().rsplit(".", 1)[-1]
        try:
            if ext == "pdf":
                pages = ocr.extract_pdf_pages(path)
            elif ext in ("md", "txt"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    pages = [{"page": 1, "text": f.read(), "extraction": "native",
                              "ocr_conf": None, "ocr_engine": None,
                              "escalated_because": None, "line_fonts": None}]
            elif ext in _UNSTRUCTURED_EXT:
                text = (_read_docx(path) if ext in ("docx", "doc")
                        else _read_unstructured(path, ext))
                if not text:
                    report["skipped"].append(
                        {"file": fname, "reason": f"no reader produced text for .{ext}"})
                    continue
                pages = [{"page": 1, "text": text, "extraction": "native",
                          "ocr_conf": None, "ocr_engine": None,
                          "escalated_because": None, "line_fonts": None}]
            else:
                report["skipped"].append({"file": fname, "reason": f"unsupported .{ext}"})
                continue
        except Exception as e:
            report["skipped"].append({"file": fname, "reason": f"parse error: {e}"})
            continue

        file_chunks, seq = [], 0
        for p in pages:
            report["pages"][p["extraction"]] += 1
            if p.get("ocr_engine") == "mistral-ocr":
                report["tier2_pages"] += 1
            if not p["text"].strip():
                continue
            new = chunker.chunk_page(p, fname, seq_start=seq)
            seq += len(new)
            file_chunks.extend(new)

        # FR-1.7: figures as first-class retrievable assets (PDFs only)
        if ext == "pdf":
            try:
                for fg in ocr.extract_page_figures(path, storage_dir):
                    file_chunks.append({
                        "id": hashlib.sha1(f"{fname}|fig{fg['figure_n']}".encode()).hexdigest()[:12],
                        "seq": seq, "source_file": fname, "page": fg["page"],
                        "extraction": "figure", "ocr_conf": fg["ocr_conf"],
                        "ocr_engine": "figure-ocr", "content_type": "figure",
                        "language": None, "figure_n": fg["figure_n"],
                        "image_path": fg["image_path"], "bbox": fg["bbox"],
                        "text": (fg["caption"] + "\n" + fg["ocr_text"]).strip()})
                    seq += 1
            except Exception as e:
                report["skipped"].append({"file": f"{fname} (figures)", "reason": str(e)[:80]})

        kept = []
        for c in file_chunks:
            h = hashlib.sha1(_norm(c["text"]).encode()).hexdigest()
            if h in seen_hashes:
                report["deduped"] += 1
                continue
            seen_hashes.add(h)
            c["namespace"] = namespace
            kept.append(c)
        chunks.extend(kept)
        report["files"].append({
            "file": fname, "pages": len(pages), "chunks": len(kept),
            "extraction": sorted({p["extraction"] for p in pages}),
            "escalations": [p["escalated_because"] for p in pages if p.get("escalated_because")],
        })

    out_path = os.path.join(storage_dir, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=1, ensure_ascii=False)
    report["total_chunks"] = len(chunks)
    report["chunks_path"] = out_path
    with open(os.path.join(storage_dir, "ingest_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return chunks, report


def main():
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else None
    chunks, report = ingest_corpus(corpus_dir)
    print(f"\n=== INGEST REPORT ===")
    print(f"pages: {report['pages']}  tier2: {report['tier2_pages']}  "
          f"deduped: {report['deduped']}  chunks: {report['total_chunks']}")
    for f in report["files"]:
        esc = f" escalations={f['escalations']}" if f["escalations"] else ""
        print(f"  {f['file']}: {f['pages']}p -> {f['chunks']} chunks {f['extraction']}{esc}")
    for s in report["skipped"]:
        print(f"  SKIPPED {s['file']}: {s['reason']}")
    print(f"\n=== SAMPLE CHUNKS (provenance proof) ===")
    shown_types = set()
    for c in chunks:
        key = (c["content_type"], c["extraction"])
        if key in shown_types:
            continue
        shown_types.add(key)
        tag = f"[{c['source_file']} p{c['page']} {c['extraction']}"
        if c["ocr_conf"] is not None:
            tag += f" ocr={c['ocr_conf']:.0%} via {c['ocr_engine']}"
        tag += f" {c['content_type']}"
        if c["language"]:
            tag += f":{c['language']}"
        tag += "]"
        print(f"\n{tag}\n{c['text'][:220]}...")


if __name__ == "__main__":
    main()
