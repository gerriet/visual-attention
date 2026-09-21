# Paper B. `make` builds main.pdf; `make final` hides the draft's todo/source notes.
TEX = pdflatex -interaction=nonstopmode -halt-on-error

main.pdf: main.tex references.bib
	$(TEX) main.tex >/dev/null
	bibtex main >/dev/null
	$(TEX) main.tex >/dev/null
	$(TEX) main.tex | grep -E "Warning: (Citation|Reference)|undefined|Overfull \\\\hbox \(([2-9][0-9]|[0-9]{3})" || true

final:
	sed 's/\\drafttrue/\\draftfalse/' main.tex > main_final.tex
	$(TEX) main_final.tex >/dev/null && bibtex main_final >/dev/null && $(TEX) main_final.tex >/dev/null && $(TEX) main_final.tex >/dev/null

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc main_final.* main.pdf

.PHONY: final clean
