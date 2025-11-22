#!/usr/bin/env python3
"""
generate_lm.py — Collect corpus from datasets, clean text based on tokenizer, build an ARPA LM using KenLM (lmplz), and optionally generate a binary LM with build_binary.

Usage examples:
  python scripts/generate_lm.py --order 4 --out lm/smartnotes_4gram.arpa
  python scripts/generate_lm.py --order 4 --out lm/smartnotes_4gram.arpa --bin lm/smartnotes_4gram.bin

Requirements:
  - kenlm `lmplz` and `build_binary` in PATH (see https://github.com/kpu/kenlm)
  - (Optional) pyctcdecode for LM integration in inference pipeline
"""

from __future__ import annotations

import argparse
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader.ocr_dataloader import TextTokenizer
from src.dataloader.ocr_dataloader import clean_text as dl_clean_text
from config import Config


def find_exe(name: str) -> Optional[str]:
    return shutil.which(name)


def collect_corpus(root_dir: str, limit: Optional[int] = None) -> List[str]:
    """Collect text lines from datasets: CensusHWR TSVs, IAM ascii, GNHK JSONs, manifest files.

    Returns a list of lines (strings).
    """
    root = Path(root_dir)
    lines: List[str] = []

    # 1) CensusHWR TSVs
    census_dir = root / "CensusHWR"
    if census_dir.exists():
        for tsv in [census_dir / p for p in ["train.tsv", "val.tsv", "test.tsv"]]:
            if tsv.exists():
                with tsv.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            lines.append(parts[1].strip())
                            if limit and len(lines) >= limit:
                                return lines

    # 2) IAM lines
    iam_ascii = root / "IAM" / "ascii" / "lines.txt"
    iam_lines_dir = root / "IAM" / "lines"
    if iam_ascii.exists():
        try:
            with iam_ascii.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[1] == "ok":
                        text = " ".join(parts[8:])
                        lines.append(text.strip())
                        if limit and len(lines) >= limit:
                            return lines
        except Exception:
            pass

    # 3) GNHK JSONs
    gnhk_dir = root / "GNHK" / "test"
    if gnhk_dir.exists():
        import json
        for json_file in gnhk_dir.glob("*.json"):
            try:
                meta = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    t = meta.get("transcription") or meta.get("text") or ""
                    if t:
                        lines.append(t)
                elif isinstance(meta, list):
                    for item in meta:
                        t = item.get("transcription") or item.get("text") or ""
                        if t:
                            lines.append(t)
                if limit and len(lines) >= limit:
                    return lines
            except Exception:
                continue

    # 4) handwritten/printed manifest files
    for manifest_name in ["handwritten_notes_manifest.txt", "printed_notes_manifest.txt"]:
        mpath = root / manifest_name
        if mpath.exists():
            with mpath.open("r", encoding="utf-8") as f:
                for l in f:
                    parts = l.strip().split("\t")
                    if len(parts) == 2:
                        lines.append(parts[1].strip())
                        if limit and len(lines) >= limit:
                            return lines

    # 5) Optionally collect texts from results file
    results_file = PROJECT_ROOT / "results" / "ocr_val_predictions.csv"
    if results_file.exists():
        import csv
        with results_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Ground Truth" in row:
                    lines.append(row["Ground Truth"].strip())
                    if limit and len(lines) >= limit:
                        return lines

    return lines


def clean_corpus_lines(lines: List[str], lowercase: bool = True) -> List[str]:
    """Clean lines using tokenizer — remove unwanted characters and normalize whitespace."""
    tokenizer = TextTokenizer()
    cleaned: List[str] = []
    allowed = set(tokenizer.chars + " ")  # allow spaces

    for l in lines:
        if lowercase:
            l = l.lower()
        # Replace tabs/newlines with spaces and trim
        l = l.replace("\t", " ").replace("\n", " ").strip()
        # Keep only allowed characters (map others to space)
        # This ensures LM and tokenizer charset align
        l2 = "".join(ch if ch in allowed else " " for ch in l)
        # Collapse repeated spaces
        l2 = " ".join(p for p in l2.split(" ") if p != "") if l2.strip() != "" else ""
        if l2:
            cleaned.append(l2)
    return cleaned


def write_lines_to_file(lines: List[str], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l.strip() + "\n")


def run_lmplz(input_file: Path, arpa_out: Path, order: int = 4, prune: Optional[str] = None) -> None:
    lmplz = find_exe("lmplz")
    if lmplz is None:
        raise RuntimeError("`lmplz` not found in PATH. Install kenlm (https://github.com/kpu/kenlm).")

    cmd = [lmplz, "-o", str(order)]
    if prune:
        cmd += ["--prune"]
        # Note: using --prune without arguments is supported, but we accept a simple string if supplied
    print("Running: ", " ".join(cmd))
    with input_file.open("r", encoding="utf-8") as fin, arpa_out.open("w", encoding="utf-8") as fout:
        subprocess.run(cmd, stdin=fin, stdout=fout, check=True)


def run_build_binary(arpa_file: Path, bin_out: Path) -> None:
    build_bin = find_exe("build_binary")
    if build_bin is None:
        raise RuntimeError("`build_binary` not found in PATH. Install kenlm (https://github.com/kpu/kenlm).")
    subprocess.run([build_bin, str(arpa_file), str(bin_out)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LM from repo datasets using KenLM")
    parser.add_argument("--root", type=str, default=Config.dataset.ROOT_DIR, help="Datasets root")
    parser.add_argument("--order", type=int, default=4, help="N-gram order")
    parser.add_argument("--out", type=str, default=str(Path(PROJECT_ROOT / "lm") / f"smartnotes_{4}gram.arpa"), help="Output ARPA path")
    parser.add_argument("--bin", type=str, default=None, help="Optional binary LM output path (.bin)")
    parser.add_argument("--build-binary", action="store_true", default=False, help="Build binary LM after creating ARPA")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Only collect/clean corpus and write corpus file, do not call lmplz or build_binary")
    parser.add_argument("--lowercase", action="store_true", default=True, help="Lowercase corpus before building LM")
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="Max number of lines to include (for quick tests)")
    parser.add_argument("--debug", action="store_true", help="Debug/print additional info")
    args = parser.parse_args()

    root = Path(args.root)
    out_arpa = Path(args.out)
    out_bin = Path(args.bin) if args.bin else None

    print(f"Collecting corpus from {root}...")
    lines = collect_corpus(str(root), limit=args.limit)
    if not lines:
        print("No lines found in datasets. Exiting.")
        sys.exit(1)

    if args.debug:
        print(f"Collected {len(lines)} lines (first 5):")
        for l in lines[:5]:
            print(repr(l))

    cleaned = clean_corpus_lines(lines, lowercase=args.lowercase)
    if args.debug:
        print(f"After cleaning {len(cleaned)} lines (first 5):")
        for l in cleaned[:5]:
            print(repr(l))

    tmp_corpus = PROJECT_ROOT / "lm" / "smartnotes_corpus.txt"
    write_lines_to_file(cleaned, tmp_corpus)
    print(f"Wrote cleaned corpus to: {tmp_corpus}")

    # Build ARPA with lmplz
    if args.dry_run:
        print("Dry run: skipped building ARPA/BN. Corpus saved at:", tmp_corpus)
        sys.exit(0)

    print(f"Building {args.order}-gram LM... (this may take a while)")
    try:
        run_lmplz(tmp_corpus, out_arpa, order=args.order)
        print(f"ARPA saved to: {out_arpa}")
    except Exception as e:
        print("Failed to generate ARPA LM:", e)
        sys.exit(2)

    # Optionally build binary
    if args.build_binary or out_bin:
        if out_bin is None:
            out_bin = out_arpa.with_suffix(".bin")
        try:
            run_build_binary(out_arpa, out_bin)
            print(f"Binary LM saved to: {out_bin}")
        except Exception as e:
            print("Failed to build binary LM:", e)
            sys.exit(3)

    print("Done.")


if __name__ == "__main__":
    main()
