# Build the paper:      make -C paper
# arXiv source bundle:  make -C paper arxiv   (writes paper/arxiv.tar.gz)
# Regenerate figures:   make -C paper figures (needs `uv sync --extra plots`)
PY ?= uv run --extra plots python

main.pdf: main.tex refs.bib neurips_2026.sty figures/frontier.pdf figures/foresight.pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	cd .. && $(PY) scripts/make_frontier_figure.py && $(PY) scripts/make_foresight_figure.py

arxiv: main.pdf
	rm -rf arxiv && mkdir -p arxiv/figures
	cp main.tex main.bbl neurips_2026.sty arxiv/
	cp figures/frontier.pdf figures/foresight.pdf arxiv/figures/
	cd arxiv && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && rm -f main.aux main.log main.out main.pdf
	COPYFILE_DISABLE=1 tar -czf arxiv.tar.gz -C arxiv .
	@echo "wrote paper/arxiv.tar.gz"

clean:
	latexmk -C
	rm -rf arxiv arxiv.tar.gz

.PHONY: figures arxiv clean
