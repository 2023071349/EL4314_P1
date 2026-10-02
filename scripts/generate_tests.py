#!/usr/bin/env python3
"""Genera entradas binarias: entero int32 LE y N números float32 LE."""

import argparse
import random
import struct
from pathlib import Path


DEFAULT_SIZES = (0, 1, 7, 8, 15, 16, 1000)
LARGE_SIZES = (100_000, 1_000_000, 50_000_000)
BLOCK_SIZE = 65_536


def write_case(path, count, value_at):
    """Escribe por bloques para mantener acotado el uso de memoria."""
    with path.open("wb") as output:
        output.write(struct.pack("<i", count))
        for start in range(0, count, BLOCK_SIZE):
            length = min(BLOCK_SIZE, count - start)
            values = [value_at(index) for index in range(start, start + length)]
            output.write(struct.pack(f"<{length}f", *values))


def large_value(index):
    # Cada grupo suma 1 exactamente, sin crecer 50 millones de veces.
    # Los enteros y las sumas parciales son representables en float32.
    # La media es 1/16 y la suma final permanece por debajo de 2**24.
    position = index % 16
    return -3.0 if position == 0 else 4.0 if position == 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Directorio de salida (predeterminado: data).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Semilla reproducible (predeterminada: 2026).")
    parser.add_argument("--large", action="store_true",
                        help="Incluye 100000, 1000000 y 50000000 elementos.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = random.Random(args.seed)

    for count in DEFAULT_SIZES:
        # Múltiplos binarios exactos, positivos y con media alejada de cero.
        values = [generator.randrange(1, 33) / 8.0 for _ in range(count)]
        path = args.output_dir / f"input_{count}.dat"
        write_case(path, count, values.__getitem__)
        print(f"{path}: {count} elementos")

    cases = {
        "negative": [-float(index + 1) for index in range(16)],
        "constant": [2.5] * 16,
        # Incluso la suma de sus cuadrados queda dentro del rango float32.
        "extremes": [-1e18, 1e18, 1e18, 1e18] * 4,
    }
    for name, values in cases.items():
        path = args.output_dir / f"input_{name}.dat"
        write_case(path, len(values), values.__getitem__)
        print(f"{path}: {len(values)} elementos")

    if args.large:
        for count in LARGE_SIZES:
            path = args.output_dir / f"input_{count}.dat"
            write_case(path, count, large_value)
            print(f"{path}: {count} elementos")


if __name__ == "__main__":
    main()
