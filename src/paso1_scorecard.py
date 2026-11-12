"""Paso 1. Scorecard clásico de originación sobre la base completa (todos los
solicitantes con resultado conocido): WoE/IV, regresión logística, escala de
puntos, Gini/KS/AUC en prueba, PSI, curva de aprobación y corte propuesto."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from comun import (CANDIDATAS, COSTO_BUENO, COSTO_MALO, OBJETIVO, SALIDA, SEMILLA,
                   SENSIBLES, Scorecard, WoE, cargar, dividir, factor_y_base, metricas, psi)


def corte_por_costo(puntos: np.ndarray, malo: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Recorre cortes y devuelve el que minimiza el costo (5 por cada malo aprobado,
    1 por cada bueno rechazado). Aprueba quien tiene puntaje >= corte."""
    cortes = np.unique(np.round(puntos)).astype(float)
    filas = []
    for c in cortes:
        aprob = puntos >= c
        malos_aprob = int((malo[aprob] == 1).sum())
        buenos_rech = int(((malo == 0) & ~aprob).sum())
        filas.append({"corte": c, "tasa_aprobacion": aprob.mean(),
                      "tasa_malos_aprobados": malos_aprob / max(aprob.sum(), 1),
                      "costo": COSTO_MALO * malos_aprob + COSTO_BUENO * buenos_rech})
    t = pd.DataFrame(filas)
    return float(t.loc[t["costo"].idxmin(), "corte"]), t


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    df = cargar()
    dev, prueba = dividir(df, SEMILLA)
    y_dev, y_pr = dev[OBJETIVO].to_numpy(), prueba[OBJETIVO].to_numpy()

    # IV de todas las candidatas y de las variables sensibles (solo informativo)
    iv_cand = WoE(CANDIDATAS).fit(dev, y_dev).iv_
    iv_sens = WoE(SENSIBLES, continuas=["edad"]).fit(dev, y_dev).iv_
    iv = pd.concat([iv_cand.rename("iv"), iv_sens.rename("iv")]).to_frame()
    iv["entra_al_modelo"] = [v in CANDIDATAS and iv_cand[v] >= 0.02 for v in iv.index]
    iv["sensible_excluida"] = iv.index.isin(SENSIBLES)
    iv.sort_values("iv", ascending=False).to_csv(SALIDA / "01_iv_variables.csv", index_label="variable")

    sc = Scorecard().fit(dev, y_dev)
    tabla = sc.tabla_puntos()
    tabla.to_csv(SALIDA / "01_scorecard_tabla.csv", index=False)

    p_dev, p_pr = sc.pd_(dev), sc.pd_(prueba)
    s_dev, s_pr = sc.puntos(dev), sc.puntos(prueba)
    m_dev, m_pr = metricas(y_dev, p_dev), metricas(y_pr, p_pr)
    # Referencia: el mismo scorecard agregando estado civil/sexo y extranjero (la edad no se probó), para ver
    # cuánto poder predictivo se cede al excluirlas.
    sc_s = Scorecard(variables=CANDIDATAS + ["estado_civil_sexo", "extranjero"]).fit(dev, y_dev)
    m_pr_s = metricas(y_pr, sc_s.pd_(prueba))
    # validación cruzada del desarrollo, para fijar el corte sin mirar la prueba
    oof = np.zeros(len(dev))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(dev, y_dev):
        oof[b] = Scorecard().fit(dev.iloc[a], y_dev[a]).puntos(dev.iloc[b])
    corte, curva_dev = corte_por_costo(oof, y_dev)
    _, curva_pr = corte_por_costo(s_pr, y_pr)
    aprob = s_pr >= corte
    sc_psi = psi(s_dev, s_pr)

    curva_dev.to_csv(SALIDA / "01_curva_aprobacion_desarrollo_cv.csv", index=False)
    curva_pr.to_csv(SALIDA / "01_curva_aprobacion_prueba.csv", index=False)

    factor, base = factor_y_base()
    resumen = {
        "n_desarrollo": len(dev), "n_prueba": len(prueba),
        "tasa_malos_desarrollo": float(y_dev.mean()), "tasa_malos_prueba": float(y_pr.mean()),
        "variables_en_modelo": sc.seleccion_,
        "escala": {"puntos_base": 600, "momios_base": "50 a 1", "pdo": 20,
                   "factor": factor, "offset": base},
        "desarrollo": m_dev, "prueba": m_pr,
        "prueba_con_sensibles": m_pr_s,
        "psi_desarrollo_vs_prueba": sc_psi,
        "corte_propuesto": corte,
        "aprobacion_prueba_en_corte": float(aprob.mean()),
        "tasa_malos_aprobados_prueba": float(y_pr[aprob].mean()),
        "tasa_malos_rechazados_prueba": float(y_pr[~aprob].mean()),
        "costo_prueba_en_corte": float(COSTO_MALO * ((y_pr == 1) & aprob).sum()
                                       + COSTO_BUENO * ((y_pr == 0) & ~aprob).sum()),
        "costo_prueba_aprobar_todos": float(COSTO_MALO * y_pr.sum()),
    }
    (SALIDA / "01_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    # --- gráficas
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ivs = iv.sort_values("iv")
    colores = ["#c0504d" if s else ("#4f81bd" if e else "#bbbbbb")
               for s, e in zip(ivs["sensible_excluida"], ivs["entra_al_modelo"])]
    ax[0].barh(ivs.index, ivs["iv"], color=colores)
    ax[0].axvline(0.02, ls="--", c="k", lw=0.8)
    ax[0].set_title("IV por variable\n(rojo: sensible, excluida)")
    ax[0].tick_params(axis="y", labelsize=8)
    bins = np.linspace(min(s_pr), max(s_pr), 20)
    ax[1].hist(s_pr[y_pr == 0], bins, alpha=0.6, label="buenos", color="#4f81bd")
    ax[1].hist(s_pr[y_pr == 1], bins, alpha=0.6, label="malos", color="#c0504d")
    ax[1].axvline(corte, ls="--", c="k")
    ax[1].set_title(f"Puntaje en prueba (corte {corte:.0f})")
    ax[1].set_xlabel("puntos")
    ax[1].legend()
    ax[2].plot(curva_pr["tasa_aprobacion"], curva_pr["tasa_malos_aprobados"], c="#4f81bd")
    ax[2].scatter([aprob.mean()], [y_pr[aprob].mean()], c="k", zorder=3, label="corte propuesto")
    ax[2].set_xlabel("tasa de aprobación")
    ax[2].set_ylabel("tasa de malos entre aprobados")
    ax[2].set_title("Curva de aprobación vs. incumplimiento (prueba)")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(SALIDA / "01_scorecard.png", dpi=130)
    plt.close()

    print(f"Variables en el modelo ({len(sc.seleccion_)}): {', '.join(sc.seleccion_)}")
    print(f"Prueba: AUC {m_pr['auc']:.3f}  Gini {m_pr['gini']:.3f}  KS {m_pr['ks']:.3f}")
    print(f"Con variables sensibles: AUC {m_pr_s['auc']:.3f}")
    print(f"PSI desarrollo vs prueba: {sc_psi:.3f}")
    print(f"Corte {corte:.0f}: aprueba {aprob.mean():.1%}, malos entre aprobados {y_pr[aprob].mean():.1%}")


if __name__ == "__main__":
    main()
