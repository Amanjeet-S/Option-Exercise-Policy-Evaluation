.PHONY: all clean verify
all: methods.pdf

methods.pdf: main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=methods main.tex

clean:
	latexmk -c -jobname=methods main.tex

verify:
	python3 verification/conditional_drift.py
	python3 verification/allocation_probability.py
	python3 verification/numerical_results.py
	python3 -m unittest discover -s verification -v
