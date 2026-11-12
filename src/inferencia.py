"""Experimento de inferencia de rechazados con rechazo SIMULADO.

La base de German Credit no trae rechazados: todos los solicitantes tienen
resultado. Para estudiar el sesgo de selección se simula una política histórica
(ver `regla_historica` en comun.py) que rechaza al peor X% y se oculta el resultado
de los rechazados al entrenar. Como el resultado real sí se conoce, se puede medir
cuánto sesgo introduce el truncamiento y cuánto recupera cada método."""
import numpy as np
import pandas as pd

from comun import OBJETIVO, Scorecard, auc_seguro, dividir, ks, regla_historica

METODOS = ["oraculo", "solo_aprobados", "augmentation", "parcelamiento", "fuzzy"]
ETIQUETAS = {
    "oraculo": "Oráculo (ve todos los resultados)",
    "solo_aprobados": "Solo aprobados (sesgado)",
    "augmentation": "Augmentation (reponderación)",
    "parcelamiento": "Parcelamiento",
    "fuzzy": "Fuzzy augmentation",
}
FACTOR_MALOS = 2.0  # supuesto: los rechazados tienen el doble de malos que predice el modelo de aprobados
N_BANDAS = 10


def simular_rechazo(desarrollo: pd.DataFrame, prueba: pd.DataFrame, rechazo: float,
                    fuga: float, rng: np.random.Generator):
    """Aplica la política histórica simulada. Rechaza el `rechazo` más alto del puntaje
    de la regla en desarrollo. En prueba se marca quién habría sido rechazado con el
    mismo corte (esa población sí tiene resultado y sirve para evaluar)."""
    r_dev = regla_historica(desarrollo, rng, fuga)
    r_pr = regla_historica(prueba, rng, fuga)
    corte = np.quantile(r_dev, 1 - rechazo)
    return r_dev > corte, r_pr > corte


def _pesos_augmentation(p_todos: np.ndarray, aprobado: np.ndarray) -> np.ndarray:
    """Cada aprobado pesa (solicitantes de su banda de score) / (aprobados de su banda)."""
    cortes = np.unique(np.quantile(p_todos, np.linspace(0, 1, N_BANDAS + 1)[1:-1]))
    banda = np.searchsorted(cortes, p_todos, side="right")
    k = len(cortes) + 1
    total = np.bincount(banda, minlength=k).astype(float)
    aprob = np.bincount(banda[aprobado], minlength=k).astype(float)
    factor = np.where(aprob > 0, total / np.maximum(aprob, 1), 0.0)
    return factor[banda[aprobado]]


def entrenar_metodos(dev: pd.DataFrame, rechazado: np.ndarray, rng: np.random.Generator,
                     factor_malos: float = FACTOR_MALOS, metodos=METODOS) -> dict:
    """Devuelve un scorecard por método. Los rechazados solo aportan su resultado en el
    oráculo; los demás métodos no lo usan."""
    y = dev[OBJETIVO].to_numpy()
    aprob = ~rechazado
    mods = {}
    if "oraculo" in metodos:
        mods["oraculo"] = Scorecard().fit(dev, y)
    sesgado = Scorecard().fit(dev[aprob], y[aprob])
    if "solo_aprobados" in metodos:
        mods["solo_aprobados"] = sesgado
    p_todos = sesgado.pd_(dev)

    if "augmentation" in metodos:
        w = _pesos_augmentation(p_todos, aprob)
        mods["augmentation"] = Scorecard().fit(dev[aprob], y[aprob], w)

    p_rej = np.clip(factor_malos * p_todos[rechazado], 0.0, 0.95)
    rej = dev[rechazado]
    if "parcelamiento" in metodos:
        # Parcelas por score entre los rechazados; en cada una se reparten etiquetas
        # malo/bueno al azar con la tasa de malos inflada.
        etiqueta = np.zeros(len(rej))
        orden = np.argsort(p_rej)
        for parcela in np.array_split(orden, 5):
            if len(parcela) == 0:
                continue
            n_malos = int(round(p_rej[parcela].mean() * len(parcela)))
            marcas = np.zeros(len(parcela))
            marcas[:n_malos] = 1
            rng.shuffle(marcas)
            etiqueta[parcela] = marcas
        X = pd.concat([dev[aprob], rej], ignore_index=True)
        mods["parcelamiento"] = Scorecard().fit(X, np.concatenate([y[aprob], etiqueta]))
    if "fuzzy" in metodos:
        X = pd.concat([dev[aprob], rej, rej], ignore_index=True)
        yy = np.concatenate([y[aprob], np.ones(len(rej)), np.zeros(len(rej))])
        w = np.concatenate([np.ones(aprob.sum()), p_rej, 1 - p_rej])
        mods["fuzzy"] = Scorecard().fit(X, yy, w)
    return mods


def evaluar(mods: dict, prueba: pd.DataFrame, rech_prueba: np.ndarray) -> pd.DataFrame:
    y = prueba[OBJETIVO].to_numpy()
    filas = []
    for nombre, m in mods.items():
        p = m.pd_(prueba)
        fila = {"metodo": nombre, "auc": auc_seguro(y, p), "ks": ks(y, p),
                "auc_zona_rechazada": auc_seguro(y[rech_prueba], p[rech_prueba]),
                "auc_zona_aprobada": auc_seguro(y[~rech_prueba], p[~rech_prueba]),
                "pd_media": float(p.mean()), "tasa_real": float(y.mean()),
                "error_calibracion": float(p.mean() - y.mean()),
                "brier": float(np.mean((p - y) ** 2))}
        fila["gini"] = 2 * fila["auc"] - 1
        filas.append(fila)
    return pd.DataFrame(filas)


def replica(df: pd.DataFrame, semilla: int, rechazo: float = 0.30, fuga: float = 0.0,
            factor_malos: float = FACTOR_MALOS, metodos=METODOS) -> pd.DataFrame:
    """Una repetición completa: nueva división, nuevo rechazo, ajuste y evaluación."""
    rng = np.random.default_rng(semilla)
    dev, prueba = dividir(df, semilla)
    rech_dev, rech_pr = simular_rechazo(dev, prueba, rechazo, fuga, rng)
    mods = entrenar_metodos(dev, rech_dev, rng, factor_malos, metodos)
    res = evaluar(mods, prueba, rech_pr)
    res["semilla"] = semilla
    res["rechazo"], res["fuga"], res["factor_malos"] = rechazo, fuga, factor_malos
    return res


def recuperacion(m: np.ndarray, sesgado: np.ndarray, oraculo: np.ndarray) -> float:
    """Fracción de la brecha entre el modelo sesgado y el oráculo que cierra el método
    (1 = igual al oráculo, 0 = igual al sesgado). Se calcula con promedios de réplicas."""
    brecha = np.mean(oraculo) - np.mean(sesgado)
    return float("nan") if abs(brecha) < 1e-12 else float((np.mean(m) - np.mean(sesgado)) / brecha)
