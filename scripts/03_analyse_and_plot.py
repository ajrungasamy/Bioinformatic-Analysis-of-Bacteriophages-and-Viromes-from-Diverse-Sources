#!/usr/bin/env python3
"""
Script 3 – Statistical summaries and figure generation.

Description
This script performs statistical summarisation and visualisation of genome-level
and protein-level data produced by the previous steps in the pipeline.

Workflow
1. Load the joined genome dataset produced during metadata integration.
2. Calculate derived metrics including gene density and hypothetical protein fraction.
3. Apply transparent quality-control flags to identify records suitable for analysis.
4. Generate descriptive statistics for genome-level features.
5. Calculate non-parametric correlations between key variables.
6. Produce figures illustrating genome architecture patterns across the dataset.

Inputs
- output/joined/phage_genomes_joined.tsv
- output/joined/phage_proteins_joined.tsv (optional)

Outputs
- Statistical summaries written to output/stats/
- Figures written to output/plots/ and output/plots_qc/

Notes
The script is designed for large bacteriophage genome datasets and uses
density-based visualisations and non-parametric statistics to summarise
patterns in genome organisation and annotation completeness.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

RANDOM_SEED = 0
MAX_SCATTER_POINTS = 20000
HEXBIN_THRESHOLD = 50000
MAX_PER_PHAGE = 60
MAX_STACKED = 40

# --------------------------- helpers ---------------------------

def ensure_dirs(base: Path) -> Dict[str, Path]:
    d = {
        "stats": base / "stats",
        "qc": base / "qc",
        "genome": base / "genome",
        "combined": base / "combined",
        "lifestyle": base / "lifestyle",
        "phrog": base / "phrog",
        "per_phage": base / "per_phage",
        "stratified": base / "stratified",
        "host": base / "host",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(outdir: Path, stem: str):
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}.png", dpi=300)
    plt.savefig(outdir / f"{stem}.pdf", dpi=300)
    plt.close()
    print(f"[FIGURE] Saved: {outdir / (stem + '.png')} (+ PDF)")


def safe_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def normalize_lifestyle(series: pd.Series) -> pd.Series:
    s = series.fillna("unknown").astype(str).str.strip().str.lower()
    return s.replace({"nan": "unknown", "": "unknown"})


def choose_key(df: pd.DataFrame) -> str:
    return "accession" if "accession" in df.columns else "phage_name"


def coalesce_merge_suffixes(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    """
    If df has base_col_x and base_col_y, create base_col and drop suffix cols.
    Prefers _x unless it's blank/NA, then falls back to _y.
    """
    x = f"{base_col}_x"
    y = f"{base_col}_y"

    if base_col in df.columns:
        return df

    if x in df.columns and y in df.columns:
        dx = df[x]
        dy = df[y]
        out = dx.copy()

        mask = out.isna() | (out.astype(str).str.strip() == "")
        out.loc[mask] = dy.loc[mask]

        df[base_col] = out
        df = df.drop(columns=[x, y])
        return df

    if x in df.columns and base_col not in df.columns:
        df = df.rename(columns={x: base_col})
    if y in df.columns and base_col not in df.columns:
        df = df.rename(columns={y: base_col})
    return df


def write_note(outdir: Path, filename: str, lines: List[str]):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / filename).write_text("\n".join(lines), encoding="utf-8")


# --------------------------- QC + stratification ---------------------------

def add_qc_flags(
    df: pd.DataFrame,
    min_len_bp: int = 2000,
    min_genes: int = 10,
    max_len_bp: int = 500_000,
    max_density_genes_per_kb: float = 3.0,
) -> pd.DataFrame:
    """
    Adds:
      qc_pass (bool)
      qc_reason (string; 'pass' or semicolon-separated reasons)
    """
    df["qc_pass"] = True
    df["qc_reason"] = ""

    rules: List[Tuple[str, pd.Series]] = [
        ("missing_length", df["genome_length"].isna() | (df["genome_length"] <= 0)),
        ("too_short", df["genome_length"].notna() & (df["genome_length"] < min_len_bp)),
        ("missing_genes", df["num_genes"].isna()),
        ("low_cds", df["num_genes"].notna() & (df["num_genes"] < min_genes)),
        ("zero_or_nan_density", df["gene_density_genes_per_kb"].isna() | (df["gene_density_genes_per_kb"] <= 0)),
        ("extreme_length", df["genome_length"].notna() & (df["genome_length"] > max_len_bp)),
        ("extreme_density", df["gene_density_genes_per_kb"].notna() & (df["gene_density_genes_per_kb"] > max_density_genes_per_kb)),
    ]

    for name, mask in rules:
        if mask.any():
            df.loc[mask, "qc_pass"] = False
            df.loc[mask, "qc_reason"] = np.where(
                df.loc[mask, "qc_reason"] == "", name, df.loc[mask, "qc_reason"] + ";" + name
            )

    df["qc_reason"] = df["qc_reason"].replace("", "pass")
    return df


def add_size_classes(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 20_000, 80_000, 200_000, np.inf]
    labels = ["small(<20kb)", "medium(20-80kb)", "large(80-200kb)", "giant(>200kb)"]
    df["genome_size_class"] = pd.cut(df["genome_length"], bins=bins, labels=labels, include_lowest=True)
    return df


def write_qc_report(df: pd.DataFrame, outdir: Path, thresholds: dict):
    outdir.mkdir(parents=True, exist_ok=True)

    # counts
    total = len(df)
    pass_n = int(df["qc_pass"].sum()) if "qc_pass" in df.columns else 0
    fail_n = total - pass_n

    # reason counts: split semicolon lists
    reason_series = df["qc_reason"].fillna("pass").astype(str)
    exploded = reason_series.str.split(";").explode()
    counts = exploded.value_counts().rename_axis("qc_reason").reset_index(name="count")
    counts["fraction"] = counts["count"] / total

    counts.to_csv(outdir / "qc_reason_counts.tsv", sep="\t", index=False)

    lines = [
        "QC SUMMARY",
        f"Total rows: {total:,}",
        f"QC-pass rows: {pass_n:,} ({pass_n/total:.2%})",
        f"QC-fail rows: {fail_n:,} ({fail_n/total:.2%})",
        "",
        "QC THRESHOLDS",
    ]
    for k, v in thresholds.items():
        lines.append(f"{k}: {v}")
    lines += [
        "",
        "TOP QC REASONS (by count)",
        counts.head(20).to_string(index=False),
    ]
    (outdir / "qc_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # bar chart
    top = counts[counts["qc_reason"] != "pass"].head(12)
    if not top.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(top["qc_reason"], top["count"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Count")
        plt.title("QC fail reasons (top 12)")
        save_fig(outdir, "bar_qc_reason_counts_top12")


# --------------------------- load + prep ---------------------------

def load_genomes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # fix common “_x/_y” merge suffix mess
    df = coalesce_merge_suffixes(df, "phage_name")
    df = coalesce_merge_suffixes(df, "accession")
    df = coalesce_merge_suffixes(df, "host")
    df = coalesce_merge_suffixes(df, "lifestyle")

    safe_numeric(df, ["genome_length", "num_genes", "num_trnas", "num_proteins", "n_hypotheticals"])

    if "lifestyle" in df.columns:
        df["lifestyle"] = normalize_lifestyle(df["lifestyle"])
    else:
        df["lifestyle"] = "unknown"

    if "host" not in df.columns:
        df["host"] = np.nan

    if {"n_hypotheticals", "num_proteins"}.issubset(df.columns):
        df["hypo_fraction"] = df["n_hypotheticals"] / df["num_proteins"].replace(0, np.nan)
    else:
        df["hypo_fraction"] = np.nan

    if {"num_genes", "genome_length"}.issubset(df.columns):
        df["gene_density_genes_per_kb"] = df["num_genes"] / df["genome_length"].replace(0, np.nan) * 1000.0
    else:
        df["gene_density_genes_per_kb"] = np.nan

    # log columns for better hexbins
    df["log10_genome_length"] = np.log10(df["genome_length"].replace(0, np.nan))
    df["log10_num_genes"] = np.log10(df["num_genes"].replace(0, np.nan))

    # genome size classes for stratified comparisons
    df = add_size_classes(df)

    # QC flags (thresholds tuned for phage-scale sanity; you can edit later)
    df = add_qc_flags(df, min_len_bp=2000, min_genes=10, max_len_bp=500_000, max_density_genes_per_kb=3.0)

    return df


def phrog_category_fractions(proteins_path: Path, key: str) -> Optional[pd.DataFrame]:
    if not proteins_path.exists():
        return None

    dfp = pd.read_csv(proteins_path, sep="\t", low_memory=False)
    if dfp.empty:
        return None

    dfp = coalesce_merge_suffixes(dfp, key)
    dfp = coalesce_merge_suffixes(dfp, "phrog_category_code")

    if key not in dfp.columns or "phrog_category_code" not in dfp.columns:
        return None

    dfp[key] = dfp[key].astype(str)
    dfp["_code"] = dfp["phrog_category_code"].fillna("").astype(str)

    total = dfp.groupby(key).size().rename("n_proteins").reset_index()

    wanted = ["S", "s", "R", "A", "T", "H", "P", "F"]
    counts = (
        dfp[dfp["_code"].isin(wanted)]
        .groupby([key, "_code"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    out = total.merge(counts, on=key, how="left").fillna(0)

    for c in wanted:
        if c not in out.columns:
            out[c] = 0
        out[f"frac_{c}"] = out[c] / out["n_proteins"].replace(0, np.nan)

    out["frac_structural_total"] = (out.get("S", 0) + out.get("s", 0)) / out["n_proteins"].replace(0, np.nan)
    out["frac_unknown_F"] = out.get("F", 0) / out["n_proteins"].replace(0, np.nan)
    return out


# --------------------------- stats ---------------------------

def write_stats(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return

    desc = numeric.describe().T
    desc.index.name = "metric"
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc["range"] = desc["max"] - desc["min"]
    desc.to_csv(outdir / "global_stats.tsv", sep="\t")

    lines = []
    for feat, row in desc.iterrows():
        lines.append(
            f"{feat}: n={int(row['count'])}, min={row['min']:.3g}, Q1={row['25%']:.3g}, "
            f"median={row['50%']:.3g}, mean={row['mean']:.3g}, Q3={row['75%']:.3g}, "
            f"max={row['max']:.3g}, IQR={row['IQR']:.3g}, range={row['range']:.3g}"
        )
    (outdir / "global_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    if "lifestyle" in df.columns:
        by = df.groupby("lifestyle")[numeric.columns].agg(["count", "mean", "median", "std"])
        by.to_csv(outdir / "by_lifestyle.tsv", sep="\t")


def spearman_corrs(df: pd.DataFrame, outdir: Path):
    cols = [c for c in ["genome_length", "num_genes", "hypo_fraction", "gene_density_genes_per_kb"] if c in df.columns]
    if len(cols) < 2:
        return
    d = df[cols].dropna()
    if len(d) < 5:
        return
    corr = d.corr(method="spearman")
    corr.to_csv(outdir / "spearman_correlations.tsv", sep="\t")


def plot_spearman_heatmap(df: pd.DataFrame, outdir: Path, stem: str = "heatmap_spearman"):
    cols = [c for c in ["genome_length", "num_genes", "hypo_fraction", "gene_density_genes_per_kb"] if c in df.columns]
    d = df[cols].dropna()
    if len(d) < 10:
        return
    corr = d.corr(method="spearman")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(im, label="Spearman ρ")
    plt.xticks(range(len(cols)), cols, rotation=30, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Spearman correlations (QC-passed)")
    save_fig(outdir, stem)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    max_n = 10000
    rng = np.random.default_rng(0)
    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    if len(y) > max_n:
        y = rng.choice(y, size=max_n, replace=False)

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def write_effect_size(df: pd.DataFrame, outdir: Path):
    if "lifestyle" not in df.columns or "hypo_fraction" not in df.columns:
        return

    d = df[["lifestyle", "hypo_fraction"]].dropna()
    if d.empty or d["lifestyle"].nunique() < 2:
        return

    sizes = d["lifestyle"].value_counts()
    g1, g2 = sizes.index[0], sizes.index[1]

    x = d.loc[d["lifestyle"] == g1, "hypo_fraction"].to_numpy(dtype=float)
    y = d.loc[d["lifestyle"] == g2, "hypo_fraction"].to_numpy(dtype=float)

    delta = cliffs_delta(x, y)

    lines = [
        f"Effect size (Cliff's delta) for hypo_fraction: {g1} vs {g2}",
        f"n({g1})={len(x):,}  n({g2})={len(y):,}",
        f"Cliff's delta = {delta:.4f}",
        "Interpretation: >0 means first group tends to have higher values."
    ]
    (outdir / "effect_size_cliffs_delta.txt").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(outdir: Path, genomes: Path, proteins: Path, n: int, label: str):
    text = [
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Run label: {label}",
        f"Genomes input: {genomes.resolve()}",
        f"Proteins input: {proteins.resolve() if proteins.exists() else 'NOT FOUND'}",
        f"Number of phages (rows in genomes table): {n:,}",
    ]
    (outdir / "README.txt").write_text("\n".join(text), encoding="utf-8")


# --------------------------- plots ---------------------------

def hist(df: pd.DataFrame, col: str, outdir: Path, stem: str, title: str, xlabel: str):
    if col not in df.columns:
        return
    d = df[col].dropna()
    if d.empty:
        return
    plt.figure()
    plt.hist(d, bins="auto", edgecolor="black", alpha=0.85)
    plt.axvline(d.median(), linestyle="--", label=f"Median: {d.median():.3g}")
    plt.xlabel(xlabel)
    plt.ylabel("Number of phages")
    plt.title(title)
    plt.legend()
    save_fig(outdir, stem)


def scatter_or_hex(df: pd.DataFrame, x: str, y: str, outdir: Path, stem: str, title: str, xlabel: str, ylabel: str):
    if x not in df.columns or y not in df.columns:
        return
    d = df[[x, y]].dropna()
    if d.empty:
        return

    if len(d) >= HEXBIN_THRESHOLD:
        plt.figure()
        plt.hexbin(d[x], d[y], gridsize=60, mincnt=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title + " (hexbin density)")
        save_fig(outdir, stem + "_hexbin")
        return

    if len(d) > MAX_SCATTER_POINTS:
        d = d.sample(n=MAX_SCATTER_POINTS, random_state=RANDOM_SEED)

    plt.figure()
    plt.scatter(d[x], d[y], alpha=0.5, s=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(outdir, stem + "_scatter")


def hexbin_log_density(df: pd.DataFrame, x: str, y: str, outdir: Path, stem: str, title: str, xlabel: str, ylabel: str):
    d = df[[x, y]].dropna()
    if d.empty:
        return
    plt.figure()
    plt.hexbin(d[x], d[y], gridsize=70, mincnt=1, bins="log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + " (hexbin, log10 density)")
    cb = plt.colorbar()
    cb.set_label("log10(count)")
    save_fig(outdir, stem)


def violin_by_group(df: pd.DataFrame, group_col: str, value_col: str, outdir: Path, stem: str, ylabel: str, title: str,
                    max_groups: int = 12) -> bool:
    if group_col not in df.columns or value_col not in df.columns:
        return False
    d = df[[group_col, value_col]].dropna()
    if d.empty:
        return False
    if d[group_col].nunique() < 2:
        return False

    groups = [(g, sub[value_col].values) for g, sub in d.groupby(group_col)]
    groups.sort(key=lambda t: len(t[1]), reverse=True)
    groups = groups[:max_groups]

    labels = [g[0] for g in groups]
    data = [g[1] for g in groups]

    plt.figure(figsize=(max(6, 0.65 * len(labels)), 5))
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), [str(x) for x in labels], rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(outdir, stem)
    return True


def binned_median_iqr(df: pd.DataFrame, x: str, y: str, outdir: Path, stem: str, title: str, xlabel: str, ylabel: str,
                      n_bins: int = 30):
    d = df[[x, y]].dropna()
    if len(d) < 100:
        return
    d = d.sort_values(x)
    q = min(n_bins, d[x].nunique())
    if q < 5:
        return
    d["bin"] = pd.qcut(d[x], q=q, duplicates="drop")
    g = d.groupby("bin")[y]
    med = g.median()
    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)
    x_mid = d.groupby("bin")[x].median()

    plt.figure()
    plt.plot(x_mid, med)
    plt.fill_between(x_mid, q1, q3, alpha=0.25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + " (median ± IQR by bins)")
    save_fig(outdir, stem)


def stacked_phrog_by_size_class(df: pd.DataFrame, outdir: Path, stem: str):
    needed = ["frac_structural_total", "frac_R", "frac_A", "frac_unknown_F"]
    if any(c not in df.columns for c in needed):
        return
    if "genome_size_class" not in df.columns:
        return

    d = df[df["genome_size_class"].notna()][["genome_size_class"] + needed].dropna()
    if d.empty:
        return

    g = d.groupby("genome_size_class")[needed].median()

    plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(g))
    labels = [str(x) for x in g.index.tolist()]
    for col in needed:
        vals = g[col].values
        plt.bar(labels, vals, bottom=bottom, label=col)
        bottom += np.nan_to_num(vals)

    plt.ylabel("Median fraction")
    plt.title("PHROG functional allocation by genome size class (median fractions)")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    save_fig(outdir, stem)


# --------------------------- run ---------------------------

def run(genomes: Path, proteins: Path, outdir: Path, clean: bool):
    # Keep your original behaviour for output/plots
    if clean and outdir.exists():
        ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        old = outdir.with_name(outdir.name + f"_old_{ts}")
        outdir.rename(old)
        print(f"[INFO] Moved old plots to: {old}")

    # New QC output folder (never overwrites your original plots folder)
    outdir_qc = outdir.parent / (outdir.name + "_qc")

    dirs = ensure_dirs(outdir)
    dirs_qc = ensure_dirs(outdir_qc)

    print(f"[INFO] Loading genomes: {genomes}")
    df = load_genomes(genomes)
    n = len(df)
    print(f"[INFO] Genome rows (phages): {n:,}")

    key = choose_key(df)

    phrog = phrog_category_fractions(proteins, key=key)
    if phrog is not None and key in df.columns:
        df[key] = df[key].astype(str)
        phrog[key] = phrog[key].astype(str)
        df = df.merge(phrog, on=key, how="left")
        print(f"[INFO] Added PHROG category fractions for {phrog.shape[0]:,} phages")
    else:
        print("[INFO] No PHROG composition added (protein table missing or incompatible)")

    # QC reports
    thresholds = {
        "min_len_bp": 2000,
        "min_genes": 10,
        "max_len_bp": 500_000,
        "max_density_genes_per_kb": 3.0,
    }
    write_qc_report(df, dirs_qc["qc"], thresholds)

    # Split into ALL vs QC-PASS
    df_all = df.copy()
    df_qc_pass = df[df["qc_pass"]].copy()

    # Write stats (both)
    write_stats(df_all, dirs["stats"])
    spearman_corrs(df_all, dirs["stats"])
    write_effect_size(df_all, dirs["stats"])
    write_manifest(outdir, genomes, proteins, n, label="ALL_ROWS")

    write_stats(df_qc_pass, dirs_qc["stats"])
    spearman_corrs(df_qc_pass, dirs_qc["stats"])
    plot_spearman_heatmap(df_qc_pass, dirs_qc["stats"], stem="heatmap_spearman_qc_pass")
    write_manifest(outdir_qc, genomes, proteins, len(df_qc_pass), label="QC_PASS_ONLY")

    # ---------------- Existing plots (UNCHANGED) ----------------
    if n > 1:
        hist(df_all, "genome_length", dirs["genome"], "hist_genome_length", "Genome length distribution", "Genome length (bp)")
        hist(df_all, "num_genes", dirs["genome"], "hist_num_genes", "Gene count distribution", "Number of genes (CDS)")
        hist(df_all, "hypo_fraction", dirs["genome"], "hist_hypo_fraction", "Hypothetical fraction distribution", "Hypothetical fraction")
        hist(df_all, "gene_density_genes_per_kb", dirs["genome"], "hist_gene_density", "Gene density distribution", "Gene density (genes/kb)")

        scatter_or_hex(df_all, "genome_length", "num_genes", dirs["combined"],
                       "genome_length_vs_num_genes", "Genome length vs gene count",
                       "Genome length (bp)", "Number of genes (CDS)")

        scatter_or_hex(df_all, "genome_length", "hypo_fraction", dirs["combined"],
                       "genome_length_vs_hypo_fraction", "Genome length vs hypothetical fraction",
                       "Genome length (bp)", "Hypothetical fraction")

        scatter_or_hex(df_all, "genome_length", "gene_density_genes_per_kb", dirs["combined"],
                       "genome_length_vs_gene_density", "Genome length vs gene density",
                       "Genome length (bp)", "Gene density (genes/kb)")

        # Lifestyle violins (unchanged logic)
        produced_any = False
        produced_any |= violin_by_group(df_all, "lifestyle", "genome_length", dirs["lifestyle"],
                                        "violin_genome_length_by_lifestyle",
                                        "Genome length (bp)", "Genome length by lifestyle")
        produced_any |= violin_by_group(df_all, "lifestyle", "hypo_fraction", dirs["lifestyle"],
                                        "violin_hypo_fraction_by_lifestyle",
                                        "Hypothetical fraction", "Hypothetical fraction by lifestyle")
        produced_any |= violin_by_group(df_all, "lifestyle", "gene_density_genes_per_kb", dirs["lifestyle"],
                                        "violin_gene_density_by_lifestyle",
                                        "Gene density (genes/kb)", "Gene density by lifestyle")

        for col, ylabel, title in [
            ("frac_structural_total", "Structural fraction (S+s)", "Structural proteins by lifestyle"),
            ("frac_unknown_F", "Unknown fraction (F)", "Unknown proteins by lifestyle"),
            ("frac_R", "Replication fraction (R)", "Replication proteins by lifestyle"),
            ("frac_A", "Assembly fraction (A)", "Assembly proteins by lifestyle"),
        ]:
            if col in df_all.columns:
                produced_any |= violin_by_group(df_all, "lifestyle", col, dirs["phrog"], f"violin_{col}_by_lifestyle", ylabel, title)

        if not produced_any:
            vc = df_all["lifestyle"].value_counts(dropna=False)
            write_note(
                dirs["lifestyle"],
                "VIOLINS_NOT_PRODUCED.txt",
                [
                    "No violin plots were produced because lifestyle has <2 groups.",
                    f"Unique lifestyles = {df_all['lifestyle'].nunique()}",
                    "Top lifestyle counts:",
                    vc.head(20).to_string(),
                    "",
                    "Fix: fill in input/metadata.csv with at least two lifestyle labels (e.g., lytic, temperate).",
                ],
            )

    # ---------------- New dissertation-grade plots (QC-passed) ----------------
    if len(df_qc_pass) > 1:
        # QC-passed distributions
        hist(df_qc_pass, "genome_length", dirs_qc["genome"], "hist_genome_length_qc_pass", "Genome length distribution (QC-passed)", "Genome length (bp)")
        hist(df_qc_pass, "num_genes", dirs_qc["genome"], "hist_num_genes_qc_pass", "Gene count distribution (QC-passed)", "Number of genes (CDS)")
        hist(df_qc_pass, "hypo_fraction", dirs_qc["genome"], "hist_hypo_fraction_qc_pass", "Hypothetical fraction distribution (QC-passed)", "Hypothetical fraction")
        hist(df_qc_pass, "gene_density_genes_per_kb", dirs_qc["genome"], "hist_gene_density_qc_pass", "Gene density distribution (QC-passed)", "Gene density (genes/kb)")

        # Readable hexbins with log-density + log x/y columns
        hexbin_log_density(df_qc_pass, "log10_genome_length", "log10_num_genes", dirs_qc["combined"],
                           "hexbin_log10_genome_length_vs_log10_num_genes_qc_pass",
                           "Genome length vs gene count (QC-passed)", "log10 genome length (bp)", "log10 number of genes")

        hexbin_log_density(df_qc_pass, "log10_genome_length", "gene_density_genes_per_kb", dirs_qc["combined"],
                           "hexbin_log10_genome_length_vs_gene_density_qc_pass",
                           "Genome length vs gene density (QC-passed)", "log10 genome length (bp)", "Gene density (genes/kb)")

        hexbin_log_density(df_qc_pass, "log10_genome_length", "hypo_fraction", dirs_qc["combined"],
                           "hexbin_log10_genome_length_vs_hypo_fraction_qc_pass",
                           "Genome length vs hypothetical fraction (QC-passed)", "log10 genome length (bp)", "Hypothetical fraction")

        # Binned median±IQR trends (very easy to write about)
        binned_median_iqr(df_qc_pass, "genome_length", "num_genes", dirs_qc["stratified"],
                          "trend_num_genes_vs_genome_length_binned_median_iqr_qc_pass",
                          "Gene content scales with genome length", "Genome length (bp)", "Number of genes (CDS)")

        binned_median_iqr(df_qc_pass, "genome_length", "hypo_fraction", dirs_qc["stratified"],
                          "trend_hypo_fraction_vs_genome_length_binned_median_iqr_qc_pass",
                          "Annotation completeness vs genome length", "Genome length (bp)", "Hypothetical fraction")

        binned_median_iqr(df_qc_pass, "genome_length", "gene_density_genes_per_kb", dirs_qc["stratified"],
                          "trend_gene_density_vs_genome_length_binned_median_iqr_qc_pass",
                          "Genome packing vs genome length", "Genome length (bp)", "Gene density (genes/kb)")

        # Size-class violins (works even when lifestyle is all unknown)
        violin_by_group(df_qc_pass, "genome_size_class", "hypo_fraction", dirs_qc["stratified"],
                        "violin_hypo_fraction_by_genome_size_class_qc_pass",
                        "Hypothetical fraction", "Hypothetical fraction by genome size class (QC-passed)")

        violin_by_group(df_qc_pass, "genome_size_class", "gene_density_genes_per_kb", dirs_qc["stratified"],
                        "violin_gene_density_by_genome_size_class_qc_pass",
                        "Gene density (genes/kb)", "Gene density by genome size class (QC-passed)")

        # PHROG allocation by size class (stacked; extremely dissertation-friendly)
        stacked_phrog_by_size_class(df_qc_pass, dirs_qc["phrog"], "stacked_phrog_allocation_by_genome_size_class_qc_pass")

        # Host-based comparisons (top 15 hosts; optional but often strong)
        if "host" in df_qc_pass.columns:
            host_counts = df_qc_pass["host"].dropna().astype(str).value_counts()
            top_hosts = host_counts.head(15).index.tolist()
            dhost = df_qc_pass[df_qc_pass["host"].astype(str).isin(top_hosts)].copy()
            if len(dhost) > 0 and dhost["host"].nunique() >= 2:
                violin_by_group(dhost, "host", "hypo_fraction", dirs_qc["host"],
                                "violin_hypo_fraction_by_host_top15_qc_pass",
                                "Hypothetical fraction", "Hypothetical fraction by host (top 15, QC-passed)", max_groups=15)

                violin_by_group(dhost, "host", "gene_density_genes_per_kb", dirs_qc["host"],
                                "violin_gene_density_by_host_top15_qc_pass",
                                "Gene density (genes/kb)", "Gene density by host (top 15, QC-passed)", max_groups=15)

    print("[DONE] Analysis complete.")
    print(f"[INFO] Plots folder (ALL): {outdir.resolve()}")
    print(f"[INFO] Plots folder (QC):  {outdir_qc.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes", type=Path, default=Path("output/joined/phage_genomes_joined.tsv"))
    ap.add_argument("--proteins", type=Path, default=Path("output/joined/phage_proteins_joined.tsv"))
    ap.add_argument("--out", type=Path, default=Path("output/plots"))
    ap.add_argument("--clean", action="store_true", help="Move old plots folder aside before writing new.")
    args = ap.parse_args()
    run(args.genomes, args.proteins, args.out, clean=args.clean)
