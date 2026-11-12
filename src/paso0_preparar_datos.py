"""Paso 0. Descarga la base pública y la convierte a CSV con nombres en español.

Fuente: UCI Machine Learning Repository, "Statlog (German Credit Data)"
(Hofmann, 1994). Son 1,000 solicitudes de crédito de un banco alemán, cada una
con su resultado (bueno o malo). No contiene datos de ningún banco mexicano."""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
# Suma SHA-256 del zip original, calculada la primera vez que se descargó.
# Si UCI lo cambia, el paso se detiene para revisarlo.
SHA256 = "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b"
MIEMBRO = "german.data"

COLUMNAS = [
    "chequera", "duracion", "historial", "proposito", "monto", "ahorros",
    "empleo", "tasa_cuota", "estado_civil_sexo", "otros_deudores", "residencia",
    "propiedad", "edad", "otros_planes", "vivienda", "creditos_existentes",
    "trabajo", "personas_cargo", "telefono", "extranjero", "clase",
]


def main() -> None:
    carpeta = RAIZ / "datos"
    carpeta.mkdir(exist_ok=True)
    destino = carpeta / "uci_german.zip"
    if not destino.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            destino.write_bytes(r.read())
    if hashlib.sha256(destino.read_bytes()).hexdigest() != SHA256:
        raise SystemExit("La suma de verificación del archivo descargado no coincide. "
                         "Revisa que sea la base original antes de continuar.")
    with zipfile.ZipFile(destino) as z:
        z.extract(MIEMBRO, carpeta)  # solo el archivo esperado
    df = pd.read_csv(carpeta / MIEMBRO, sep=r"\s+", header=None, names=COLUMNAS)
    if df.shape != (1000, 21) or set(df["clase"]) != {1, 2}:
        raise SystemExit("La estructura de la base no es la esperada (1,000 filas, 21 columnas).")
    df["malo"] = (df["clase"] == 2).astype(int)  # en la base original 2 = malo
    df = df.drop(columns="clase")
    df.to_csv(carpeta / "german_credit.csv", index=False)
    print(f"Base lista: {df.shape[0]:,} solicitudes, tasa de malos {df['malo'].mean():.1%}")


if __name__ == "__main__":
    main()
