#!/usr/bin/env python3
"""
===========================================================
 Script 1 – Local GenBank/EMBL Parser (BIG FILE SAFE)
 Author: Ajay Rungasamy (updated)

What it does:
  • Reads GenBank (.gb/.gbk/.gbff) or EMBL (.embl), including .gz
  • Accepts: single file, directory, or .txt list of filenames
  • Streams output to disk (safe for 50,000+ records)
  • Adds stable identifier: accession = record.id

Outputs:
  1) output/summaries/phage_summary.tsv
  2) output/summaries/phage_proteins_phrog.tsv (+ .csv)  [optional]
  3) output/summaries/per_phage/<file>.tsv               [optional]

Fast modes:
  --no-per-phage   skip per_phage outputs (faster)
  --no-proteins    skip protein table (FASTEST)
===========================================================
"""

import os
import csv
import shutil
import datetime
import gzip
import re
from pathlib import Path
from typing import Optional, Tuple

from Bio import SeqIO

# ------------------- FOLDERS -------------------
for folder in [
    "input",
    "output",
    "output/summaries",
    "output/summaries/per_phage",
    "output/backups",
    "scripts",
]:
    os.makedirs(folder, exist_ok=True)

# ------------------- PHROG CATEGORY MAP -------------------
PHROG_CATEGORY_MAP = {
    "S": "Structure (curated)",
    "s": "Structure (predicted)",
    "R": "Replication",
    "A": "Assembly/packaging",
    "T": "Transcription",
    "H": "Host interaction",
    "P": "Protein metabolism",
    "F": "Function unknown",
}
PHROG_ID_RE = re.compile(r"phrog_(\d+)", re.IGNORECASE)
PHROG_CODE_RE = re.compile(r"\b([SsRrAaTtHhPpFf])\b")

# ------------------- BACKUPS -------------------
def make_backup(file_path: str):
    if os.path.exists(file_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_name = f"{Path(file_path).stem}.{timestamp}.bak"
        backup_path = Path("output") / "backups" / backup_name
        shutil.copy2(file_path, backup_path)
        print(f"[backup created] {backup_path}")

# ------------------- INPUT DISCOVERY -------------------
def read_list_from_file(list_path: Path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def get_files(input_path: str):
    p = Path(input_path)

    if p.suffix.lower() == ".txt":
        names = read_list_from_file(p)
        return [Path("input") / name for name in names]

    if p.is_file():
        return [p]

    if p.is_dir():
        files = []
        for f in p.iterdir():
            name = f.name.lower()
            if name.endswith((".gb", ".gbk", ".gbff", ".embl", ".gb.gz", ".gbk.gz", ".gbff.gz", ".embl.gz")):
                files.append(f)
        if not files:
            raise FileNotFoundError("No GenBank or EMBL files found in directory.")
        return sorted(files)

    raise FileNotFoundError(f"Input path {input_path} not found.")

def detect_format(filepath: Path) -> str:
    name = filepath.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return "embl" if name.endswith(".embl") else "genbank"

# ------------------- SAFE OPEN (TEXT MODE) -------------------
def open_seq_handle_text(filepath: Path) -> Tuple[object, str]:
    fmt = detect_format(filepath)
    if filepath.name.lower().endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="replace", newline=""), fmt
    return open(filepath, "r", encoding="utf-8", errors="replace", newline=""), fmt

# ------------------- PHROG EXTRACTION -------------------
def extract_phrog_from_feature(feat):
    phrog_id = None
    phrog_cat = None
    phrog_func = None

    for key in ("phrog", "phrog_id", "phrog_desc", "note", "product", "function"):
        vals = feat.qualifiers.get(key)
        if not vals:
            continue
        text = " ".join(str(v) for v in vals)

        if phrog_id is None:
            m = PHROG_ID_RE.search(text)
            if m:
                phrog_id = f"phrog_{m.group(1)}"

        if phrog_cat is None:
            m = PHROG_CODE_RE.search(text)
            if m:
                phrog_cat = m.group(1)

        if phrog_func is None and key in ("phrog", "phrog_id", "phrog_desc", "product", "function"):
            phrog_func = text

        if phrog_id and phrog_cat and phrog_func:
            break

    phrog_num = None
    if phrog_id:
        try:
            phrog_num = int(phrog_id.split("_")[1])
        except Exception:
            phrog_num = None

    phrog_cat_clean = PHROG_CATEGORY_MAP.get(phrog_cat) if phrog_cat else None
    return phrog_id, phrog_num, phrog_cat, phrog_cat_clean, phrog_func

# ------------------- STREAM PARSE ONE FILE -------------------
def parse_file_streaming(
    filepath: Path,
    summary_writer: csv.DictWriter,
    proteins_writer: Optional[csv.DictWriter],
    proteins_csv_writer: Optional[csv.DictWriter],
    write_proteins: bool,
    write_per_phage: bool,
    per_phage_dir: Path,
    progress_every: int,
    limit: Optional[int],
    skip_bad_records: bool,
):
    handle, fmt = open_seq_handle_text(filepath)
    start_time = datetime.datetime.now()

    per_phage_rows = [] if write_per_phage else None
    n_records = 0
    skipped = 0

    parser = SeqIO.parse(handle, fmt)

    while True:
        try:
            rec = next(parser)
        except StopIteration:
            break
        except Exception as e:
            skipped += 1
            if not skip_bad_records:
                handle.close()
                raise
            if progress_every:
                print(f"[WARN] Skipping bad record in {filepath.name}: {e}")
            continue

        n_records += 1
        if limit is not None and n_records > limit:
            break

        if progress_every and (n_records % progress_every == 0):
            print(f"  → {filepath.name}: processed {n_records:,} records (skipped {skipped})...", flush=True)

        accession = str(rec.id) if rec.id else ""
        phage_name = str(rec.name) if rec.name else filepath.stem
        genome_length = len(rec.seq)

        gene_count = 0
        trna_count = 0
        cds_count = 0
        hypo_count = 0

        for feat in rec.features:
            t = feat.type.lower()
            if t == "gene":
                gene_count += 1
            elif t == "trna":
                trna_count += 1
            elif t == "cds":
                cds_count += 1
                product = feat.qualifiers.get("product", [""])[0]
                if "hypothetical protein" in str(product).lower():
                    hypo_count += 1

        summary_row = {
            "accession": accession,
            "phage_name": phage_name,
            "genome_length": genome_length,
            "num_genes": gene_count,
            "num_trnas": trna_count,
            "num_proteins": cds_count,
            "n_hypotheticals": hypo_count,
        }
        summary_writer.writerow(summary_row)
        if per_phage_rows is not None:
            per_phage_rows.append(summary_row)

        if write_proteins and proteins_writer and proteins_csv_writer:
            for feat in rec.features:
                if feat.type.lower() != "cds":
                    continue

                product = feat.qualifiers.get("product", [""])[0]
                protein_id = feat.qualifiers.get("protein_id", [""])[0]
                locus_tag = feat.qualifiers.get("locus_tag", [""])[0]
                phrog_id, phrog_num, phrog_cat, phrog_cat_clean, phrog_func = extract_phrog_from_feature(feat)

                prow = {
                    "accession": accession,
                    "phage_name": phage_name,
                    "genome_length": genome_length,
                    "num_genes": gene_count,
                    "num_trnas": trna_count,
                    "num_proteins": cds_count,
                    "n_hypotheticals": hypo_count,
                    "protein_id": protein_id,
                    "locus_tag": locus_tag,
                    "product": product,
                    "phrog_category_code": phrog_cat,
                    "phrog_category_clean": phrog_cat_clean,
                    "phrog_id": phrog_id,
                    "phrog_num": phrog_num,
                    "phrog_function": phrog_func,
                }
                proteins_writer.writerow(prow)
                proteins_csv_writer.writerow(prow)

    handle.close()

    if per_phage_rows is not None:
        per_phage_dir.mkdir(parents=True, exist_ok=True)
        per_path = per_phage_dir / f"{filepath.stem}.tsv"
        with open(per_path, "w", newline="", encoding="utf-8") as out_ind:
            writer = csv.DictWriter(
                out_ind,
                fieldnames=[
                    "accession", "phage_name", "genome_length", "num_genes",
                    "num_trnas", "num_proteins", "n_hypotheticals"
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(per_phage_rows)

    elapsed = datetime.datetime.now() - start_time
    print(f"  ✓ {filepath.name}: finished {n_records:,} records (skipped {skipped}) in {elapsed}.")

# ------------------- MAIN -------------------
def main(
    input_path: str,
    output_tsv: str,
    protein_tsv: str,
    no_proteins: bool,
    no_per_phage: bool,
    progress_every: int,
    limit: Optional[int],
    skip_bad_records: bool,
):
    make_backup(output_tsv)
    make_backup(protein_tsv)
    protein_csv = protein_tsv.replace(".tsv", ".csv")
    make_backup(protein_csv)

    files = get_files(input_path)
    per_phage_dir = Path("output/summaries/per_phage")

    write_proteins = not no_proteins
    write_per_phage = not no_per_phage

    if no_per_phage:
        print("[INFO] --no-per-phage enabled: skipping per_phage outputs (FASTER).")
    if no_proteins:
        print("[INFO] --no-proteins enabled: skipping protein table (FASTEST).")

    with open(output_tsv, "w", newline="", encoding="utf-8") as out_sum:
        summary_writer = csv.DictWriter(
            out_sum,
            fieldnames=[
                "accession", "phage_name", "genome_length", "num_genes",
                "num_trnas", "num_proteins", "n_hypotheticals"
            ],
            delimiter="\t",
        )
        summary_writer.writeheader()

        proteins_writer = None
        proteins_csv_writer = None
        out_p_tsv = None
        out_p_csv = None

        if write_proteins:
            protein_fields = [
                "accession",
                "phage_name",
                "genome_length",
                "num_genes",
                "num_trnas",
                "num_proteins",
                "n_hypotheticals",
                "protein_id",
                "locus_tag",
                "product",
                "phrog_category_code",
                "phrog_category_clean",
                "phrog_id",
                "phrog_num",
                "phrog_function",
            ]
            out_p_tsv = open(protein_tsv, "w", newline="", encoding="utf-8")
            out_p_csv = open(protein_csv, "w", newline="", encoding="utf-8")
            proteins_writer = csv.DictWriter(out_p_tsv, fieldnames=protein_fields, delimiter="\t")
            proteins_csv_writer = csv.DictWriter(out_p_csv, fieldnames=protein_fields, delimiter=",")
            proteins_writer.writeheader()
            proteins_csv_writer.writeheader()

        for f in files:
            print(f"[parse] {f}")
            parse_file_streaming(
                filepath=f,
                summary_writer=summary_writer,
                proteins_writer=proteins_writer,
                proteins_csv_writer=proteins_csv_writer,
                write_proteins=write_proteins,
                write_per_phage=write_per_phage,
                per_phage_dir=per_phage_dir,
                progress_every=progress_every,
                limit=limit,
                skip_bad_records=skip_bad_records,
            )

        if out_p_tsv is not None:
            out_p_tsv.close()
        if out_p_csv is not None:
            out_p_csv.close()

    print("\nScript 1 complete.")
    print(f"   Genome summary: {output_tsv}")
    if write_proteins:
        print(f"   Protein+PHROG:  {protein_tsv} (+ CSV)")
    print("   Per-file summaries:", "ENABLED" if write_per_phage else "DISABLED")

# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Summarise local GenBank/EMBL files (big-file safe).")
    ap.add_argument("input", help="Path to a GenBank/EMBL file, directory, or list.")
    ap.add_argument("--out", default="output/summaries/phage_summary.tsv", help="Genome-level summary TSV.")
    ap.add_argument("--proteins", default="output/summaries/phage_proteins_phrog.tsv", help="Protein-level PHROG TSV.")
    ap.add_argument("--no-proteins", action="store_true", help="Skip protein/PHROG table (faster).")
    ap.add_argument("--no-per-phage", action="store_true", help="Skip per_phage outputs (faster).")
    ap.add_argument("--progress-every", type=int, default=5000, help="Print progress every N records (0 disables).")
    ap.add_argument("--limit", type=int, default=None, help="Only parse first N records (testing).")
    ap.add_argument("--no-skip-bad-records", action="store_true", help="Stop on first bad record.")
    args = ap.parse_args()

    main(
        input_path=args.input,
        output_tsv=args.out,
        protein_tsv=args.proteins,
        no_proteins=args.no_proteins,
        no_per_phage=args.no_per_phage,
        progress_every=args.progress_every,
        limit=args.limit,
        skip_bad_records=(not args.no_skip_bad_records),
    )
