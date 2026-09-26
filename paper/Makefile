.PHONY: all clean
all: methods.pdf

methods.pdf: main.tex references.bib $(wildcard sections/*.tex)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=methods main.tex

clean:
	latexmk -c -jobname=methods main.tex
