# Paper (IEEEtran, conference)

- `main.tex`: the manuscript. It contains no result number: tables are `\input` from `tables/`, and the numbers quoted in
  the prose are macros from `tables/numbers.tex`.
- `tables/`, `figures/`: generated, do not edit. Rebuild with
  `python scripts/tables.py --data crowdai && python scripts/tables.py --data whu`,
  `python scripts/figures.py --data crowdai --results --qualitative`, then `python scripts/paper_tables.py`, which also
  checks that every number, table, figure and citation used by `main.tex` exists.
- `refs.bib`: references, checked against Crossref or the arXiv abstract page.

Compile with any TeX distribution that has IEEEtran, booktabs and TikZ (TeX Live, MiKTeX, Overleaf):

    pdflatex main && bibtex main && pdflatex main && pdflatex main

The title in `\papertitle` is a working title in English; the registered title of the course project is in `CLAUDE.md`.
