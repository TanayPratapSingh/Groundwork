.PHONY: help install data index eval eval-full demo test serve gate clean check

PYTHON ?= python3

help:
	@echo "Groundwork"
	@echo ""
	@echo "  make install    install dependencies"
	@echo "  make data       download the real datasets (~7 MB)"
	@echo "  make index      build the retrieval index"
	@echo "  make eval       run the evaluation and the release gate"
	@echo "  make eval-full  evaluate all 11,873 questions"
	@echo "  make demo       walk the system end to end"
	@echo "  make test       run the test suite"
	@echo "  make serve      run the HTTP gateway"
	@echo "  make check      test + eval, the pre-push check"
	@echo "  make clean      remove build artifacts (keeps downloaded data)"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/fetch_data.py

index: data
	$(PYTHON) scripts/build_index.py

eval:
	$(PYTHON) scripts/run_eval.py

eval-full:
	$(PYTHON) scripts/run_eval.py --full

# Record the current metrics as the regression baseline. Run this
# deliberately, on a build you are happy to be held to.
gate:
	$(PYTHON) scripts/run_eval.py --set-baseline

demo:
	$(PYTHON) scripts/demo.py

test:
	$(PYTHON) -m pytest tests/ -q

serve:
	$(PYTHON) -c "import sys; sys.path.insert(0,'src'); \
		from groundwork.gateway.server import run; run()"

# What CI runs, and what to run before pushing. `eval` exits non-zero when
# the gate fails, so this fails the build on a quality regression and not
# only on a broken test.
check: test eval

clean:
	rm -rf artifacts/ data/processed/
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache
