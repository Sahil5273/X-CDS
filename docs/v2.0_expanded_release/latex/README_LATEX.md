# Compiling the LaTeX Project on Windows

This is a self-contained, journal-agnostic LaTeX project for the X-CDS research paper.

## Directory Structure
- `main.tex`           → Master document
- `references.bib`     → BibTeX bibliography database (25 entries)
- `sections/`          → Individual section source files
- `figures/`           → Flowchart diagrams (TikZ drawings)
- `tables/`            → Data and error analysis tables (booktabs)

## Method 1: Using latexmk (Recommended)
If you have a LaTeX distribution (like MiKTeX or TeX Live) and Perl installed, run:
```bash
cd docs/v2.0_expanded_release/latex
latexmk -pdf main.tex
```
To clean auxiliary files:
```bash
latexmk -c
```

## Method 2: Manual compilation (pdflatex + bibtex)
If compiling manually, run the following sequence to resolve citations and cross-references:
```bash
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
```

## Method 3: Overleaf (Cloud)
To compile online:
1. Zip the contents of the `latex/` directory (make sure `main.tex` is at the root of the zip).
2. Upload the zip file directly to Overleaf.
3. Overleaf will automatically detect `main.tex` and compile.
