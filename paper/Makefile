.PHONY: all clean verify
all: Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf

Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf: main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=Exercise_Policy_Loss_under_Computational_and_Financial_Constraints main.tex

clean:
	latexmk -c -jobname=Exercise_Policy_Loss_under_Computational_and_Financial_Constraints main.tex

verify:
	python3 verification/conditional_drift.py
	python3 verification/allocation_probability.py
	python3 verification/numerical_results.py
	python3 verification/paper_results.py
	python3 verification/cpu_timings.py
	python3 -m unittest discover -s verification -v
