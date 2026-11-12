"""Corre todos los pasos en orden. Con la misma semilla da los mismos resultados.
El paso 0 (descarga) se corre aparte; el CSV de la base ya viene en datos/."""
import subprocess
import sys
from pathlib import Path

PASOS = [
    "paso1_scorecard.py",
    "paso2_rechazados.py",
    "paso3_sensibilidad.py",
]

if __name__ == "__main__":
    carpeta = Path(__file__).resolve().parent
    for paso in PASOS:
        print(f"\n=== {paso} ===")
        subprocess.run([sys.executable, str(carpeta / paso)], check=True, cwd=carpeta)
