#!/usr/bin/env python3
"""
Script 2 – Join annotation outputs with metadata (scales to 50k+).

Purpose (keeps intended role):
  - Take Script 1 outputs (summary + protein table)
  - Ensure metadata.csv exists and contains all phage identifiers
  - Join metadata onto genomes (small table) and proteins (huge table, chunked)
  - Write clean joined TSVs (NO phage_name_x/phage_name_y mess)
  - Write a QC report so you can defend your pipeline in the dissertation

Inputs:
  output/summaries/phage_summary.tsv
  output/summaries/phage_proteins_phrog.tsv  (optional, can be huge)

Ensures:
  input/metadata.csv exists and includes all phages

Outputs:
  output/joined/phage_genomes_joined.tsv
  output/joined/phage_proteins_joined.tsv
  output/joined/join_qc.txt
"""

from __future__ import annotations

from pathlib import Path
import datetime
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def choose_key(genomes_df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in genomes_df.columns:
        return preferred
    return "accession" if "accession" in genomes_df.columns else "phage_name"


def normalize_lifestyle(s: pd.Series) -> pd.Series:
    return (
        s.fillna("unknown")
        .astype(str).str.strip().str.lower()
        .replace({"nan": "unknown", "": "unknown"})
    )


def ensure_metadata(genomes_df: pd.DataFrame, metadata_path: Path, key: str) -> pd.DataFrame:
    """
    Create/update metadata.csv so it includes ALL phages.
    Keep both accession + phage_name if available (human-edit friendly).
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if key not in genomes_df.columns:
        raise ValueError(f"Genome summary must contain '{key}' column.")

    id_cols = [c for c in ["accession", "phage_name"] if c in genomes_df.columns]
    identity = genomes_df[id_cols].copy()
    identity[key] = identity[key].astype(str)
    identity = identity.dropna(subset=[key]).drop_duplicates(subset=[key])

    if not metadata_path.exists():
        df_meta = identity.copy()
        df_meta["host"] = ""
        df_meta["lifestyle"] = "unknown"
    else:
        df_meta = load_table(metadata_path)

        # Ensure key exists in metadata
        if key not in df_meta.columns:
            df_meta[key] = ""

        # Ensure ID columns exist for readability
        for c in ["accession", "phage_name"]:
            if c in identity.columns and c not in df_meta.columns:
                df_meta[c] = ""

        # Add new phages missing from metadata
        df_meta[key] = df_meta[key].astype(str)
        existing = set(df_meta[key].astype(str))
        missing = identity[~identity[key].isin(existing)].copy()

        if not missing.empty:
            missing["host"] = ""
            missing["lifestyle"] = "unknown"
            df_meta = pd.concat([df_meta, missing], ignore_index=True)

    # Normalize lifestyle
    df_meta["lifestyle"] = normalize_lifestyle(df_meta.get("lifestyle", pd.Series(["unknown"] * len(df_meta))))

    # Keep clean column order
    ordered = [c for c in ["accession", "phage_name", "host", "lifestyle"] if c in df_meta.columns]
    rest = [c for c in df_meta.columns if c not in ordered]
    df_meta = df_meta[ordered + rest]

    df_meta.to_csv(metadata_path, index=False)
    print(f"[meta] Metadata saved/updated: {metadata_path}")
    return df_meta


def collapse_duplicate_columns(df: pd.DataFrame, cols=("phage_name", "host", "lifestyle")) -> pd.DataFrame:
    """
    If a merge produced col_x / col_y, collapse to a single clean col.
    Prefer _x unless empty, then fill from _y. Drop both _x/_y afterwards.
    """
    df = df.copy()
    for col in cols:
        x = f"{col}_x"
        y = f"{col}_y"
        if x in df.columns and y in df.columns:
            df[col] = df[x]
            empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
            df.loc[empty, col] = df.loc[empty, y]
            df.drop(columns=[x, y], inplace=True)
    return df


def chunked_protein_join(
    proteins_path: Path,
    meta: pd.DataFrame,
    out_proteins: Path,
    key: str,
    chunksize: int = 250_000,
) -> tuple[int, int]:
    """
    Chunked join to avoid loading the entire proteins table into RAM.
    Returns: (rows_written, chunks_processed)
    """
    out_proteins.parent.mkdir(parents=True, exist_ok=True)

    if not proteins_path.exists():
        print("[WARN] Protein table not found; skipping protein join.")
        return 0, 0

    meta = meta.copy()
    meta[key] = meta[key].astype(str)

    rows_written = 0
    chunks_processed = 0
    first = True

    for chunk in pd.read_csv(proteins_path, sep="\t", chunksize=chunksize, low_memory=False):
        chunks_processed += 1

        if key not in chunk.columns:
            raise ValueError(f"Protein table missing join key '{key}'. (Script 1 should output it.)")

        chunk[key] = chunk[key].astype(str)
        merged = chunk.merge(meta, on=key, how="left")

        merged.to_csv(
            out_proteins,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first
        )
        first = False
        rows_written += len(merged)

        if chunks_processed % 10 == 0:
            print(f"[join] proteins: chunks={chunks_processed:,} rows_written={rows_written:,}...")

    return rows_written, chunks_processed


def main(
    summary_path: Path = Path("output/summaries/phage_summary.tsv"),
    proteins_path: Path = Path("output/summaries/phage_proteins_phrog.tsv"),
    metadata_path: Path = Path("input/metadata.csv"),
    out_genomes: Path = Path("output/joined/phage_genomes_joined.tsv"),
    out_proteins: Path = Path("output/joined/phage_proteins_joined.tsv"),
    protein_chunksize: int = 250_000,
    key: str | None = None,
):
    out_genomes.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {summary_path}")
    df_genomes = load_table(summary_path)

    join_key = choose_key(df_genomes, preferred=key)
    print(f"[INFO] Join key: {join_key}")

    df_meta = ensure_metadata(df_genomes, metadata_path, join_key)

    # Join genomes (small enough to do normally)
    df_genomes[join_key] = df_genomes[join_key].astype(str)
    df_meta[join_key] = df_meta[join_key].astype(str)

    df_genomes_joined = df_genomes.merge(df_meta, on=join_key, how="left")
    df_genomes_joined = collapse_duplicate_columns(df_genomes_joined)
    if "lifestyle" in df_genomes_joined.columns:
        df_genomes_joined["lifestyle"] = normalize_lifestyle(df_genomes_joined["lifestyle"])

    df_genomes_joined.to_csv(out_genomes, sep="\t", index=False)

    # Join proteins (chunked)
    rows_written, chunks = chunked_protein_join(
        proteins_path=proteins_path,
        meta=df_meta,
        out_proteins=out_proteins,
        key=join_key,
        chunksize=protein_chunksize,
    )

    # QC report
    qc = []
    qc.append(f"Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}")
    qc.append(f"Join key: {join_key}")
    qc.append(f"Genomes input rows: {len(df_genomes):,}")
    qc.append(f"Genomes joined rows: {len(df_genomes_joined):,}")
    if "lifestyle" in df_genomes_joined.columns:
        qc.append(f"Unique lifestyles: {df_genomes_joined['lifestyle'].nunique():,}")
        qc.append(f"Unknown lifestyle rows: {(df_genomes_joined['lifestyle'] == 'unknown').sum():,}")
    qc.append(f"Protein rows written: {rows_written:,}")
    qc.append(f"Protein chunks processed: {chunks:,}")

    qc_path = out_genomes.parent / "join_qc.txt"
    qc_path.write_text("\n".join(qc), encoding="utf-8")

    print(f"[saved] {out_genomes}  shape={df_genomes_joined.shape}")
    if rows_written:
        print(f"[saved] {out_proteins}  rows_written={rows_written:,}")
    print(f"[qc] {qc_path}")
    print("Join complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=Path("output/summaries/phage_summary.tsv"))
    ap.add_argument("--proteins", type=Path, default=Path("output/summaries/phage_proteins_phrog.tsv"))
    ap.add_argument("--metadata", type=Path, default=Path("input/metadata.csv"))
    ap.add_argument("--out-genomes", type=Path, default=Path("output/joined/phage_genomes_joined.tsv"))
    ap.add_argument("--out-proteins", type=Path, default=Path("output/joined/phage_proteins_joined.tsv"))
    ap.add_argument("--protein-chunksize", type=int, default=250_000)
    ap.add_argument("--key", type=str, default=None, help="Preferred join key (accession or phage_name).")
    args = ap.parse_args()

    main(
        summary_path=args.summary,
        proteins_path=args.proteins,
        metadata_path=args.metadata,
        out_genomes=args.out_genomes,
        out_proteins=args.out_proteins,
        protein_chunksize=args.protein_chunksize,
        key=args.key,
    )
