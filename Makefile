PYTHON ?= python3
CC = gcc
NASM = nasm
CFLAGS = -O2 -g -Wall -Wextra -Wpedantic -Werror -fno-tree-vectorize -fno-tree-slp-vectorize

.PHONY: all help dependencies data data-large reference check clean

all: build/programa_scalar build/programa_vector

build:
	mkdir -p build

build/driver.o: src/driver.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/driver_vector.o: src/driver.c | build
	$(CC) $(CFLAGS) -DVECTOR_BUILD -c $< -o $@

build/libstats_vector.o: src/libstats_vector.asm | build
	$(NASM) -f elf64 -g -F dwarf $< -o $@

build/programa_vector: build/driver_vector.o build/libstats_vector.o
	$(CC) $^ -lm -o $@

build/libstats_scalar.o: src/libstats_scalar.asm | build
	$(NASM) -f elf64 -g -F dwarf $< -o $@

build/programa_scalar: build/driver.o build/libstats_scalar.o
	$(CC) $^ -lm -o $@

help:
	@echo 'Requiere Linux x86-64, CPU con AVX2, GCC, NASM, Make y Python 3.'
	@echo 'make dependencies: instala NumPy; make: compila ambas versiones.'
	@echo 'make data | data-large | reference | check | clean'

dependencies:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/generate_tests.py

data-large:
	$(PYTHON) scripts/generate_tests.py --large

reference: data
	$(PYTHON) scripts/verify.py data/input_16.dat --save-reference results/reference_16

check: all data
	mkdir -p results
	@set -e; for input in data/input_*.dat; do \
		case "$$input" in */input_0.dat) continue ;; esac; \
		build/programa_scalar "$$input" results/output_scalar.dat results/stats_scalar.json; \
		build/programa_vector "$$input" results/output_vector.dat results/stats_vector.json; \
		$(PYTHON) scripts/verify.py "$$input" \
			--scalar-stats results/stats_scalar.json --scalar-output results/output_scalar.dat \
			--vector-stats results/stats_vector.json --vector-output results/output_vector.dat; \
	done
	@if build/programa_scalar data/input_0.dat results/zero.dat results/zero.json; then exit 1; fi
	@if build/programa_vector data/input_0.dat results/zero.dat results/zero.json; then exit 1; fi

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; p = Path('build'); shutil.rmtree(p) if p.exists() else None"
