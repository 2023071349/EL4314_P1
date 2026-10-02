#!/usr/bin/env python3
"""Referencia float32 y comparación de resultados; no forma parte del kernel."""

import argparse
import json
from pathlib import Path
import struct
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("Falta NumPy. Ejecute: python3 -m pip install -r requirements.txt.")


def read_input(path):
    """El encabezado es un int de 4 bytes; después hay exactamente N floats."""
    with open(path, "rb") as file:
        header = file.read(4)
        if len(header) != 4:
            raise ValueError("Encabezado incompleto")
        n = struct.unpack("<i", header)[0]
        if n < 0:
            raise ValueError("N negativo")
        if Path(path).stat().st_size != 4 + 4 * n:
            raise ValueError("El tamaño del archivo no coincide con N")
        values = np.fromfile(file, dtype="<f4", count=n)
    if not np.all(np.isfinite(values)):
        raise ValueError("La entrada contiene NaN o infinito")
    return values


def reference(values):
    """Varianza poblacional: suma de (x-media)^2 dividida entre N."""
    n = len(values)
    if n == 0:
        raise ValueError("N=0: no se pueden calcular estadísticas")
    # Cada operación estadística se mantiene en precisión simple.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        total = np.sum(values, dtype=np.float32)
        mean = np.float32(total / np.float32(n))
        differences = values - mean
        variance = np.float32(np.sum(differences * differences,
                                     dtype=np.float32) / np.float32(n))
        stddev = np.float32(np.sqrt(variance))
        # Para sigma=0 se elige una salida de ceros y se evita dividir.
        output = np.zeros(n, dtype=np.float32) if stddev == 0 else differences / stddev
    stats = {"n": n, "sum": float(total), "mean": float(mean),
             "var": float(variance), "stddev": float(stddev),
             "min": float(np.min(values)), "max": float(np.max(values))}
    return stats, output


def compare(actual, expected, name):
    """Error relativo <=1e-4; si la referencia es cero, error absoluto <=1e-6."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: tamaño incorrecto")
    # La medición del error usa double; la referencia y los kernels usan float32.
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    limit = np.where(expected == 0, 1e-6, 1e-4 * np.abs(expected.astype(np.float64)))
    if not np.all(np.isfinite(actual)) or not np.all(error <= limit):
        raise ValueError(f"{name}: excede la tolerancia")


def read_results(stats_path, output_path, n):
    stats = json.loads(Path(stats_path).read_text())
    if stats.get("n") != n:
        raise ValueError("Las estadísticas tienen un N incorrecto")
    if Path(output_path).stat().st_size != n * 4:
        raise ValueError("La salida debe contener N floats sin encabezado")
    output = np.fromfile(output_path, dtype="<f4")
    return stats, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--save-reference", type=Path,
                        help="Prefijo de los archivos de referencia .json y .dat")
    for version in ("scalar", "vector"):
        parser.add_argument(f"--{version}-stats", type=Path)
        parser.add_argument(f"--{version}-output", type=Path)
    args = parser.parse_args()
    try:
        stats, output = reference(read_input(args.input))
        print(json.dumps(stats, indent=2, allow_nan=False))
        if args.save_reference:
            args.save_reference.parent.mkdir(parents=True, exist_ok=True)
            args.save_reference.with_suffix(".json").write_text(
                json.dumps(stats, indent=2, allow_nan=False) + "\n")
            output.astype("<f4").tofile(args.save_reference.with_suffix(".dat"))
        results = []
        for version in ("scalar", "vector"):
            stats_path = getattr(args, f"{version}_stats")
            output_path = getattr(args, f"{version}_output")
            if bool(stats_path) != bool(output_path):
                raise ValueError(f"Indique las estadísticas y la salida {version}")
            if stats_path:
                actual_stats, actual_output = read_results(stats_path, output_path, stats["n"])
                for key in ("sum", "mean", "var", "stddev", "min", "max"):
                    compare(actual_stats[key], stats[key], f"{version}: {key}")
                compare(actual_output, output, f"{version}: normalización")
                results.append((actual_stats, actual_output))
                print(f"PASS {version} contra referencia")
        if len(results) == 2:
            for key in ("sum", "mean", "var", "stddev", "min", "max"):
                compare(results[0][0][key], results[1][0][key], f"scalar/vector: {key}")
            compare(results[0][1], results[1][1], "scalar/vector: normalización")
            print("PASS scalar contra vector")
    except (OSError, ValueError, KeyError, TypeError, FloatingPointError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
