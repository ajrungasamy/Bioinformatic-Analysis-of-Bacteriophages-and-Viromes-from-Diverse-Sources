# Bioinformatic Analysis of Bacteriophages and Viromes from Diverse Sources

This repository contains the Python scripts used for the computational analysis performed in my undergraduate dissertation project.

The project performs large-scale comparative genomic analysis of bacteriophage genomes, including genome feature extraction, metadata integration, quality control, and statistical analysis.

## Analysis pipeline

The workflow consists of three main scripts:

### 1. Genome parsing
`01_parse_local_phage_summary.py`

Parses locally stored GenBank/EMBL genome files and extracts genome-level features including:
- genome length
- gene counts
- CDS counts
- tRNA counts
- hypothetical protein counts

Outputs a genome summary table used for downstream analysis.

---

### 2. Metadata integration
`02_join_metadata.py`

Combines the genome summary table with external metadata (e.g., host and lifestyle information) using accession identifiers or phage names as join keys.

---

### 3. Statistical analysis and figure generation
`03_analyse_and_plot.py`

Performs downstream analysis including:

- calculation of derived metrics (gene density, hypothetical protein fraction)
- quality-control filtering
- summary statistics
- correlation analysis
- automated generation of plots used in the dissertation

---

## Software requirements

The analysis was performed using:

- Python 3
- Biopython
- pandas
- NumPy
- matplotlib

## Data availability

The scripts provided in this repository were used to analyse bacteriophage genome annotations derived from publicly available sequence databases. Due to dataset size, genome input files and generated outputs are not included in this repository.
Ajay Rungasamy  
BSc Biological Sciences  
University of Leicester
