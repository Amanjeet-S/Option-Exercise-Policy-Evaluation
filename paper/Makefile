.PHONY: all clean verify
all: Assessing_Bermudan_Exercise_Policies.pdf

Assessing_Bermudan_Exercise_Policies.pdf: main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=Assessing_Bermudan_Exercise_Policies main.tex

clean:
	latexmk -c -jobname=Assessing_Bermudan_Exercise_Policies main.tex

verify:
	python3 verification/conditional_drift.py
	python3 verification/allocation_probability.py
	python3 verification/numerical_results.py
	python3 verification/paper_results.py
	python3 verification/revision_results.py
	python3 verification/cpu_timings.py
	python3 -m unittest discover -s verification -v
