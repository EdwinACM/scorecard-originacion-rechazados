"""Paso 3. Sensibilidad del experimento de rechazados (todo SIMULADO).

Varía el porcentaje rechazado por la política histórica, la fuga de información
del analista (rechazo no ignorable) y el factor de inflación de malos que usan el
parcelamiento y el fuzzy. Cada celda son 100 repeticiones completas."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import SALIDA, SEMILLA, cargar
from inferencia import ETIQUETAS, METODOS, replica

REPLICAS = 100
RECHAZOS = [0.10, 0.20, 0.30, 0.40, 0.50]
FUGAS = [0.0, 0.5, 1.0]
FACTORES = [1.0, 1.5, 2.0, 3.0]


def resumen(rs: pd.DataFrame, claves: list[str]) -> pd.DataFrame:
    g = rs.groupby(claves + ["metodo"])[["auc", "ks", "auc_zona_rechazada",
                                         "error_calibracion", "brier"]].mean().reset_index()
    return g


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    df = cargar()
    rng = np.random.default_rng(SEMILLA)
    brutos = []
    for fuga in FUGAS:
        for rech in RECHAZOS:
            for i in range(REPLICAS):
                brutos.append(replica(df, SEMILLA + i, rech, fuga))
    rs = pd.concat(brutos, ignore_index=True)
    rs.to_csv(SALIDA / "replicas_sensibilidad.csv", index=False)
    tabla = resumen(rs, ["fuga", "rechazo"])

    # sesgo del truncamiento con intervalo bootstrap sobre las repeticiones
    filas = []
    for (fuga, rech), g in rs.groupby(["fuga", "rechazo"]):
        a = g.pivot(index="semilla", columns="metodo", values="auc")
        dif = (a["solo_aprobados"] - a["oraculo"]).to_numpy()
        medias = dif[rng.integers(0, len(dif), (2000, len(dif)))].mean(axis=1)
        filas.append({"fuga": fuga, "rechazo": rech, "sesgo_auc": dif.mean(),
                      "lo": np.percentile(medias, 2.5), "hi": np.percentile(medias, 97.5)})
    sesgo = pd.DataFrame(filas)
    sesgo.to_csv(SALIDA / "03_sesgo_por_escenario.csv", index=False)
    tabla.to_csv(SALIDA / "03_sensibilidad_rechazo_fuga.csv", index=False)

    fac = pd.concat([replica(df, SEMILLA + i, 0.30, 0.0, f, ["oraculo", "solo_aprobados", "parcelamiento", "fuzzy"])
                     .assign(factor=f) for f in FACTORES for i in range(REPLICAS)], ignore_index=True)
    tfac = resumen(fac, ["factor"])
    tfac.to_csv(SALIDA / "03_sensibilidad_factor.csv", index=False)

    fig, ax = plt.subplots(1, len(FUGAS), figsize=(15, 4.2), sharey=True)
    colores = {"oraculo": "k", "solo_aprobados": "#c0504d", "augmentation": "#f79646",
               "parcelamiento": "#9bbb59", "fuzzy": "#4f81bd"}
    for a, fuga in zip(ax, FUGAS):
        for m in METODOS:
            t = tabla[(tabla["fuga"] == fuga) & (tabla["metodo"] == m)]
            a.plot(t["rechazo"] * 100, t["auc"], marker="o", label=ETIQUETAS[m], color=colores[m])
        a.set_title(f"fuga de información = {fuga}")
        a.set_xlabel("% rechazado por la política simulada")
    ax[0].set_ylabel("AUC en prueba completa")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(SALIDA / "03_sensibilidad.png", dpi=130)
    plt.close()

    pd.set_option("display.width", 200)
    print(sesgo.round(3).to_string(index=False))
    print(tabla.pivot_table(index=["fuga", "rechazo"], columns="metodo", values="auc").round(3).to_string())
    print(tfac.pivot_table(index="factor", columns="metodo", values=["auc", "error_calibracion"]).round(3).to_string())


if __name__ == "__main__":
    main()
