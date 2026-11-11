# One-command entry points. See README.md.
PY := uv run --extra dev --extra ml --extra plots python
OUT := outputs/reproduce

.PHONY: setup test reproduce reproduce-all figures paper arxiv

setup:            ## install the locked environment (CPU is enough for everything below)
	uv sync --locked --extra dev --extra ml --extra plots

test:             ## unit tests (CPU, no downloads)
	uv run --extra dev --extra ml pytest -q

reproduce:        ## rerun every offline analysis on data/ and check it against results/ (~1 min)
	mkdir -p $(OUT) && ln -sf ../../data/demand_trace_eval16.npz $(OUT)/demand_trace.npz
	$(PY) -m moe_eviction.experiments.belady_bound $(OUT)
	$(PY) -m moe_eviction.experiments.evict_policies $(OUT)
	$(PY) -m moe_eviction.experiments.belady_flat_bound data/demand_trace_eval16.npz --out $(OUT)/flat_bound.json
	$(PY) -m moe_eviction.experiments.belady_flat_bound_perstream --eval-trace data/demand_trace_eval16.npz --train-trace data/demand_trace_train48.npz --out $(OUT)/perstream.json
	$(PY) -m moe_eviction.experiments.belady_lookahead data/demand_trace_eval16.npz --out $(OUT)/lookahead.json
	$(PY) -m moe_eviction.experiments.belady_optimality_check
	$(PY) -m moe_eviction.experiments.belady_imitation $(OUT) --out $(OUT)/e8
	$(PY) scripts/check_offline.py

reproduce-all: reproduce  ## also retrain the router-feature evictors of Appendix C (~11 min on a laptop CPU)
	$(PY) -m moe_eviction.experiments.belady_imitation_e9 --train-trace data/demand_trace_train48.npz --eval-trace data/demand_trace_eval16.npz --out $(OUT)/e9
	$(PY) scripts/check_offline.py

figures:          ## regenerate paper/figures/*.pdf from results/
	$(PY) scripts/make_frontier_figure.py
	$(PY) scripts/make_foresight_figure.py

paper:            ## build paper/main.pdf
	$(MAKE) -C paper

arxiv:            ## build paper/arxiv.tar.gz for upload
	$(MAKE) -C paper arxiv
