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

The lab server has no TeX, so `.github/workflows/paper.yml` compiles the paper on GitHub at every push that changes
`paper/`. The PDF, the LaTeX log (`main.log`, `build.txt`) and the exit code land on the branch `paper-pdf`, which holds
build output only and is replaced by every build; a PDF that compiled is copied to `paper/main.pdf`:

    git fetch origin paper-pdf
    git show FETCH_HEAD:exit_code.txt          # 0 = compiled; main.log and build.txt hold the messages
    git show FETCH_HEAD:source_commit.txt      # the commit of main that was compiled
    git show FETCH_HEAD:main.pdf > paper/main.pdf

`main.pdf` is the compiled paper (8 pages). After a change of `main.tex` or of the tables it is stale until it is copied
again; the push of `main.pdf` itself does not start a build.

`\papertitle` holds the English title of the paper (decided 17 Nov 2026); the registered title of the course project, in
Indonesian, is in `CLAUDE.md` and does not change.
