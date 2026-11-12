"""Paso 2. Inferencia de rechazados con rechazo SIMULADO.

Escenario principal: la política histórica simulada rechaza al 30% con peor
puntaje de una regla simple, sin fuga de información sobre el resultado.
Parte A: una corrida con la división de semilla 42 e intervalos bootstrap sobre
la muestra de prueba. Parte B: 300 repeticiones completas (nueva división, nuevo
rechazo, nuevo ajuste) con intervalos bootstrap sobre las repeticiones."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import OBJETIVO, SALIDA, SEMILLA, auc_seguro, cargar, dividir, ks
from inferencia import (ETIQUETAS, FACTOR_MALOS, METODOS, entrenar_metodos, recuperacion,
                        replica, simular_rechazo)

RECHAZO = 0.30
FUGA = 0.0
N_REPLICAS = 300
N_BOOT = 2000
MET = ["auc", "ks", "auc_zona_rechazada", "error_calibracion", "brier"]


def ic(x, nivel=95):
    a = (100 - nivel) / 2
    return float(np.percentile(x, a)), float(np.percentile(x, 100 - a))


def parte_a(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEMILLA)
    dev, prueba = dividir(df, SEMILLA)
    rech_dev, rech_pr = simular_rechazo(dev, prueba, RECHAZO, FUGA, rng)
    y_dev, y = dev[OBJETIVO].to_numpy(), prueba[OBJETIVO].to_numpy()
    mods = entrenar_metodos(dev, rech_dev, rng)
    P = {m: mods[m].pd_(prueba) for m in METODOS}

    diag = {
        "rechazados_desarrollo": int(rech_dev.sum()),
        "tasa_malos_aprobados": float(y_dev[~rech_dev].mean()),
        "tasa_malos_rechazados_REAL": float(y_dev[rech_dev].mean()),
        "pd_media_rechazados_segun_modelo_de_aprobados": float(mods["solo_aprobados"].pd_(dev[rech_dev]).mean()),
        "tasa_malos_zona_rechazada_prueba": float(y[rech_pr].mean()),
        "n_zona_rechazada_prueba": int(rech_pr.sum()),
    }

    def calc(idx):
        yy, rr = y[idx], rech_pr[idx]
        out = {}
        for m in METODOS:
            p = P[m][idx]
            out[m] = {"auc": auc_seguro(yy, p), "ks": ks(yy, p),
                      "auc_zona_rechazada": auc_seguro(yy[rr], p[rr]),
                      "error_calibracion": float(p.mean() - yy.mean()),
                      "brier": float(np.mean((p - yy) ** 2))}
        return out

    puntual = calc(np.arange(len(y)))
    brng = np.random.default_rng(SEMILLA)
    boots = [calc(brng.integers(0, len(y), len(y))) for _ in range(N_BOOT)]
    filas = []
    for m in METODOS:
        fila = {"metodo": m}
        for k in MET:
            v = np.array([b[m][k] for b in boots])
            lo, hi = ic(v[~np.isnan(v)])
            fila.update({k: puntual[m][k], f"{k}_lo": lo, f"{k}_hi": hi})
        if m not in ("solo_aprobados",):
            d = np.array([b[m]["auc"] - b["solo_aprobados"]["auc"] for b in boots])
            fila["dif_auc_vs_sesgado"] = puntual[m]["auc"] - puntual["solo_aprobados"]["auc"]
            fila["dif_auc_vs_sesgado_lo"], fila["dif_auc_vs_sesgado_hi"] = ic(d)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA / "02_parteA_semilla42.csv", index=False)
    return {"diagnostico": diag, "tabla": tabla}


def parte_b(df: pd.DataFrame) -> pd.DataFrame:
    rs = pd.concat([replica(df, SEMILLA + i, RECHAZO, FUGA) for i in range(N_REPLICAS)],
                   ignore_index=True)
    rs.to_csv(SALIDA / "replicas_principal.csv", index=False)
    brng = np.random.default_rng(SEMILLA)
    ancho = {k: rs.pivot(index="semilla", columns="metodo", values=k)[METODOS] for k in MET}
    muestras = brng.integers(0, N_REPLICAS, (N_BOOT, N_REPLICAS))
    filas = []
    for m in METODOS:
        fila = {"metodo": m}
        for k in MET:
            a = ancho[k][m].to_numpy()
            medias = a[muestras].mean(axis=1)
            fila[k] = a.mean()
            fila[f"{k}_lo"], fila[f"{k}_hi"] = ic(medias)
        filas.append(fila)
    tabla = pd.DataFrame(filas)

    # sesgo del truncamiento (sesgado - oráculo) y recuperación de cada método
    extra = []
    for k in ["auc", "ks", "auc_zona_rechazada"]:
        A = ancho[k]
        dif = A["solo_aprobados"].to_numpy() - A["oraculo"].to_numpy()
        medias = dif[muestras].mean(axis=1)
        lo, hi = ic(medias)
        extra.append({"metrica": k, "metodo": "sesgo_truncamiento", "valor": dif.mean(), "lo": lo, "hi": hi})
        for m in ["augmentation", "parcelamiento", "fuzzy"]:
            r = [recuperacion(A[m].to_numpy()[i], A["solo_aprobados"].to_numpy()[i],
                              A["oraculo"].to_numpy()[i]) for i in muestras]
            r = np.array(r)
            lo, hi = ic(r[~np.isnan(r)])
            extra.append({"metrica": k, "metodo": m,
                          "valor": recuperacion(A[m].to_numpy(), A["solo_aprobados"].to_numpy(),
                                                A["oraculo"].to_numpy()), "lo": lo, "hi": hi})
    pd.DataFrame(extra).to_csv(SALIDA / "02_parteB_sesgo_y_recuperacion.csv", index=False)
    # diferencias pareadas contra el modelo sesgado
    pareadas = []
    for k in ["auc", "ks", "auc_zona_rechazada"]:
        A = ancho[k]
        for m in METODOS:
            if m == "solo_aprobados":
                continue
            d = A[m].to_numpy() - A["solo_aprobados"].to_numpy()
            lo, hi = ic(d[muestras].mean(axis=1))
            pareadas.append({"metrica": k, "metodo": m, "dif_media_vs_sesgado": d.mean(),
                             "lo": lo, "hi": hi, "pct_replicas_mejor": float((d > 0).mean())})
    pd.DataFrame(pareadas).to_csv(SALIDA / "02_parteB_pareadas_vs_sesgado.csv", index=False)
    tabla.to_csv(SALIDA / "02_parteB_replicas.csv", index=False)
    return tabla


def grafica(a: pd.DataFrame, b: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    et = [ETIQUETAS[m] for m in METODOS]
    for j, (k, titulo) in enumerate([("auc", "AUC en prueba completa"),
                                     ("auc_zona_rechazada", "AUC en la zona que la política rechazaría"),
                                     ("error_calibracion", "PD media predicha menos tasa real")]):
        y = np.arange(len(METODOS))
        for dat, color, desp, nombre in [(b, "#4f81bd", -0.15, f"{N_REPLICAS} repeticiones"),
                                         (a, "#c0504d", 0.15, "semilla 42 (bootstrap)")]:
            ax[j].errorbar(dat[k], y + desp, xerr=[dat[k] - dat[f"{k}_lo"], dat[f"{k}_hi"] - dat[k]],
                           fmt="o", color=color, capsize=3, label=nombre)
        ax[j].set_yticks(y)
        ax[j].set_yticklabels(et if j == 0 else [""] * len(et), fontsize=8)
        ax[j].invert_yaxis()
        ax[j].set_title(titulo, fontsize=10)
        if k == "error_calibracion":
            ax[j].axvline(0, c="k", lw=0.8)
        if j == 0:
            ax[j].legend(fontsize=8, loc="lower left")
    plt.suptitle("Rechazo SIMULADO del 30%: intervalos de 95%", fontsize=11)
    plt.tight_layout()
    plt.savefig(SALIDA / "02_rechazados.png", dpi=130)
    plt.close()


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    df = cargar()
    A = parte_a(df)
    B = parte_b(df)
    grafica(A["tabla"], B)
    (SALIDA / "02_diagnostico_semilla42.json").write_text(
        json.dumps({"rechazo": RECHAZO, "fuga": FUGA, "factor_malos": FACTOR_MALOS,
                    "replicas": N_REPLICAS, "bootstrap": N_BOOT, **A["diagnostico"]},
                   indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("Diagnóstico semilla 42:", json.dumps(A["diagnostico"], indent=1))
    print("\nParte A (semilla 42)")
    print(A["tabla"][["metodo", "auc", "auc_lo", "auc_hi", "auc_zona_rechazada", "error_calibracion"]].round(3).to_string(index=False))
    print(f"\nParte B ({N_REPLICAS} repeticiones)")
    print(B[["metodo", "auc", "auc_lo", "auc_hi", "auc_zona_rechazada", "error_calibracion", "brier"]].round(3).to_string(index=False))
    print(pd.read_csv(SALIDA / "02_parteB_sesgo_y_recuperacion.csv").round(3).to_string(index=False))
    print(pd.read_csv(SALIDA / "02_parteB_pareadas_vs_sesgado.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
