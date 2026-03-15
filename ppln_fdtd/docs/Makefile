# Makefile for building the FDTD technical note PDF
#
# Requirements:
#   - pandoc (>= 2.19)
#   - xelatex (via texlive or similar)
#   - Eisvogel template (optional, for polished output)
#
# Install Eisvogel:
#   mkdir -p ~/.local/share/pandoc/templates
#   curl -L https://github.com/Wandmalfarbe/pandoc-latex-template/releases/latest/download/Eisvogel.tar.gz | \
#     tar xz -C ~/.local/share/pandoc/templates
#
# Usage:
#   make pdf          # build PDF (uses Eisvogel if available, plain otherwise)
#   make figures      # regenerate all figures
#   make clean        # remove generated PDF
#   make all          # figures + PDF

SRC = fdtd_technical_note.md
PDF = fdtd_technical_note.pdf
FIGURES_SCRIPT = paper_figures.py

# Pandoc settings
ENGINE = xelatex
#FONT = "Libertinus Serif"
FONT = "Linux Libertine O"
MONO = "Inconsolata"
FONTSIZE = 11pt
MARGIN = 1in
HIGHLIGHT = tango

# Metadata passed to the template
TITLE = "Carrier-Resolved FDTD for SHG in PPLN"
SUBTITLE = "Development, Learnings, and Limitations"
AUTHOR = "Claude Code (Anthropic), Claude Opus 4.6 Extended (Anthropic), and Tom Soulanille"
DATE = "March 2026"

# Check if Eisvogel template is available
EISVOGEL := $(shell pandoc -D latex 2>/dev/null && \
	test -f $(HOME)/.local/share/pandoc/templates/eisvogel.latex && \
	echo yes)

# Common pandoc flags
PANDOC_FLAGS = \
	--pdf-engine=$(ENGINE) \
	-V geometry:margin=$(MARGIN) \
	-V fontsize=$(FONTSIZE) \
	-V colorlinks=true \
	-V linkcolor=NavyBlue \
	-V citecolor=NavyBlue \
	-V urlcolor=NavyBlue \
	--highlight-style=$(HIGHLIGHT) \
	-V title=$(TITLE) \
	-V subtitle=$(SUBTITLE) \
	-V author=$(AUTHOR) \
	-V date=$(DATE) \
	--resource-path=.:figures

ifeq ($(EISVOGEL),yes)
PANDOC_FLAGS += \
	--template=eisvogel \
	-V titlepage=true \
	-V titlepage-color=ffffff \
	-V titlepage-text-color=333333 \
	-V titlepage-rule-color=cc0000 \
	-V titlepage-rule-height=4 \
	-V book=false \
	-V classoption=oneside \
	-V toc=true \
	-V toc-own-page=true \
	-V mainfont=$(FONT) \
	-V monofont=$(MONO) \
	-V code-block-font-size=\\footnotesize
else
PANDOC_FLAGS += \
	-V mainfont=$(FONT) \
	-V monofont=$(MONO) \
	--toc
endif

.PHONY: all pdf figures clean check-deps

all: figures pdf

pdf: $(PDF)

$(PDF): $(SRC) $(wildcard figures/*.png)
	@echo "Building $(PDF)..."
	@command -v pandoc >/dev/null 2>&1 || \
		(echo "Error: pandoc not found. Install with: sudo apt install pandoc" && exit 1)
	@command -v $(ENGINE) >/dev/null 2>&1 || \
		(echo "Error: $(ENGINE) not found. Install with: sudo apt install texlive-xetex" && exit 1)
ifeq ($(EISVOGEL),yes)
	@echo "  Using Eisvogel template"
else
	@echo "  Eisvogel not found, using plain template"
	@echo "  (Install Eisvogel for a polished title page and TOC)"
endif
	pandoc $(SRC) -o $(PDF) $(PANDOC_FLAGS)
	@echo "  Done: $(PDF)"

figures:
	@echo "Generating figures..."
	python $(FIGURES_SCRIPT)
	@echo "  Done"

clean:
	rm -f $(PDF)

# Print dependency status
check-deps:
	@echo "Checking dependencies..."
	@command -v pandoc >/dev/null 2>&1 && \
		echo "  pandoc: $(shell pandoc --version | head -1)" || \
		echo "  pandoc: NOT FOUND"
	@command -v $(ENGINE) >/dev/null 2>&1 && \
		echo "  $(ENGINE): found" || \
		echo "  $(ENGINE): NOT FOUND (apt install texlive-xetex)"
	@fc-list | grep -qi "libertinus" && \
		echo "  $(FONT): found" || \
		echo "  $(FONT): NOT FOUND (apt install fonts-libertinus)"
	@fc-list | grep -qi "inconsolata" && \
		echo "  $(MONO): found" || \
		echo "  $(MONO): NOT FOUND (apt install fonts-inconsolata)"
ifeq ($(EISVOGEL),yes)
	@echo "  Eisvogel template: found"
else
	@echo "  Eisvogel template: not found (optional, see header of this Makefile)"
endif
