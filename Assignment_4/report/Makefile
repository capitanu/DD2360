FILENAME := appgpu21_HW4_20
loop:
	while inotifywait -e close_write $(FILENAME).tex ; do pdflatex $(FILENAME).tex; pdflatex $(FILENAME).tex; bibtex $(FILENAME); makeglossaries $(FILENAME); pdflatex $(FILENAME).tex; done	


once:
	pdflatex $(FILENAME).tex; pdflatex $(FILENAME).tex; bibtex $(FILENAME); makeglossaries $(FILENAME); pdflatex $(FILENAME).tex
