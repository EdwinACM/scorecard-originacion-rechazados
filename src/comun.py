"""Código compartido: carga, WoE/IV, scorecard, escala de puntos, métricas y la
política histórica simulada que genera los rechazados."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos" / "german_credit.csv"
SALIDA = RAIZ / "resultados"
SEMILLA = 42

# Variables sensibles: sexo y estado civil (vienen juntos en una sola columna),
# edad y condición de trabajador extranjero. No entran al modelo.
SENSIBLES = ["estado_civil_sexo", "edad", "extranjero"]
OBJETIVO = "malo"
CANDIDATAS = ["chequera", "duracion", "historial", "proposito", "monto", "ahorros",
              "empleo", "tasa_cuota", "otros_deudores", "residencia", "propiedad",
              "otros_planes", "vivienda", "creditos_existentes", "trabajo",
              "personas_cargo", "telefono"]
CONTINUAS = ["duracion", "monto"]  # el resto se trata como categórica

# Escala de puntos: 600 puntos con momios buenos:malos de 50 a 1, y cada 20
# puntos se duplican los momios.
PUNTOS_BASE, MOMIOS_BASE, PDO = 600, 50, 20
IV_MINIMO = 0.02
# Matriz de costos de la documentación de la base: aprobar a un malo cuesta 5 veces
# lo que cuesta rechazar a un bueno.
COSTO_MALO, COSTO_BUENO = 5.0, 1.0


def cargar() -> pd.DataFrame:
    return pd.read_csv(DATOS)


def dividir(df: pd.DataFrame, semilla: int = SEMILLA, prueba: float = 0.30):
    """División estratificada por resultado. Devuelve desarrollo y prueba."""
    d, t = train_test_split(df, test_size=prueba, stratify=df[OBJETIVO], random_state=semilla)
    return d.reset_index(drop=True), t.reset_index(drop=True)


# ---------------------------------------------------------------- WoE e IV

class WoE:
    """Agrupa cada variable en categorías (cuantiles para las continuas, categorías
    originales para el resto, con las pequeñas juntas en "otras") y calcula el
    WoE = ln(%buenos / %malos) con suavizado de 0.5. Acepta pesos."""

    def __init__(self, variables, continuas=CONTINUAS, min_frac=0.05, n_bins=5):
        self.variables, self.continuas = list(variables), set(continuas)
        self.min_frac, self.n_bins = min_frac, n_bins

    def _indice(self, v: str, x: pd.Series) -> np.ndarray:
        if v in self.continuas:
            return np.searchsorted(self.cortes_[v], x.to_numpy(), side="right")
        mapa, k = self.mapa_[v], len(self.etiquetas_[v])
        # una categoría que no se vio al ajustar cae en un grupo extra con WoE 0
        return np.array([mapa.get(c, k) for c in x], dtype=int)

    def fit(self, X: pd.DataFrame, y, w=None):
        y = np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        self.cortes_, self.mapa_, self.tablas_, self.etiquetas_ = {}, {}, {}, {}
        for v in self.variables:
            x = X[v]
            if v in self.continuas:
                q = np.quantile(x, np.linspace(0, 1, self.n_bins + 1)[1:-1])
                self.cortes_[v] = np.unique(q)
                cortes = self.cortes_[v]
                etiquetas = []
                for i in range(len(cortes) + 1):
                    if i == 0:
                        etiquetas.append(f"hasta {cortes[0]:,.0f}")
                    elif i == len(cortes):
                        etiquetas.append(f"más de {cortes[-1]:,.0f}")
                    else:
                        etiquetas.append(f"{cortes[i-1]:,.0f} a {cortes[i]:,.0f}")
            else:
                frec = x.value_counts()
                grandes = sorted(frec[frec / len(x) >= self.min_frac].index)
                mapa = {c: i for i, c in enumerate(grandes)}
                etiquetas = [str(c) for c in grandes]
                chicas = [c for c in frec.index if c not in mapa]
                if chicas:
                    for c in chicas:
                        mapa[c] = len(grandes)
                    etiquetas.append("otras")
                self.mapa_[v] = mapa
            self.etiquetas_[v] = etiquetas
            k = len(etiquetas)
            idx = self._indice(v, x)
            buenos = np.bincount(idx, weights=w * (1 - y), minlength=k)
            malos = np.bincount(idx, weights=w * y, minlength=k)
            pg = (buenos + 0.5) / (buenos.sum() + 0.5 * k)
            pb = (malos + 0.5) / (malos.sum() + 0.5 * k)
            woe = np.log(pg / pb)
            vacio = (buenos + malos) == 0
            woe[vacio] = 0.0
            t = pd.DataFrame({"variable": v, "grupo": etiquetas, "buenos": buenos,
                              "malos": malos, "woe": woe, "iv_parcial": (pg - pb) * woe})
            t["pct_obs"] = (t["buenos"] + t["malos"]) / (buenos.sum() + malos.sum())
            t["tasa_malos"] = t["malos"] / (t["buenos"] + t["malos"]).replace(0, np.nan)
            self.tablas_[v] = t
        self.iv_ = pd.Series({v: self.tablas_[v]["iv_parcial"].sum() for v in self.variables})
        return self

    def transformar(self, X: pd.DataFrame, variables=None) -> pd.DataFrame:
        variables = self.variables if variables is None else variables
        out = {}
        for v in variables:
            woe = np.append(self.tablas_[v]["woe"].to_numpy(), 0.0)
            idx = self._indice(v, X[v])
            out[v] = woe[np.minimum(idx, len(woe) - 1)]
        return pd.DataFrame(out, index=X.index)


class Scorecard:
    """WoE + regresión logística sobre las variables con IV suficiente."""

    def __init__(self, variables=CANDIDATAS, iv_minimo=IV_MINIMO, C=1.0):
        self.variables, self.iv_minimo, self.C = list(variables), iv_minimo, C

    def fit(self, X: pd.DataFrame, y, w=None):
        self.woe_ = WoE(self.variables).fit(X, y, w)
        self.seleccion_ = [v for v in self.variables if self.woe_.iv_[v] >= self.iv_minimo]
        Z = self.woe_.transformar(X, self.seleccion_)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000)
        self.lr_.fit(Z, np.asarray(y), sample_weight=w)
        return self

    def pd_(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.woe_.transformar(X, self.seleccion_))[:, 1]

    def puntos(self, X: pd.DataFrame) -> np.ndarray:
        return pd_a_puntos(self.pd_(X))

    def tabla_puntos(self) -> pd.DataFrame:
        """Puntos por grupo. El intercepto se reparte en partes iguales entre las variables."""
        factor, base = factor_y_base()
        n = len(self.seleccion_)
        b0 = float(self.lr_.intercept_[0])
        filas = []
        for v, coef in zip(self.seleccion_, self.lr_.coef_[0]):
            t = self.woe_.tablas_[v].copy()
            # La regresión modela ln(malos/buenos); el puntaje sube con ln(buenos/malos).
            t["coeficiente"] = coef
            t["puntos"] = base / n - factor * (coef * t["woe"] + b0 / n)
            filas.append(t)
        return pd.concat(filas, ignore_index=True)


def factor_y_base() -> tuple[float, float]:
    factor = PDO / np.log(2)
    base = PUNTOS_BASE - factor * np.log(MOMIOS_BASE)
    return float(factor), float(base)


def pd_a_puntos(pd_: np.ndarray) -> np.ndarray:
    pd_ = np.clip(pd_, 1e-6, 1 - 1e-6)
    factor, base = factor_y_base()
    return base + factor * np.log((1 - pd_) / pd_)


# ---------------------------------------------------------------- métricas

def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def auc_seguro(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def metricas(y, p) -> dict:
    a = auc_seguro(y, p)
    return {"auc": a, "gini": 2 * a - 1, "ks": ks(y, p)}


def psi(esperado: np.ndarray, real: np.ndarray, bins: int = 10) -> float:
    """Índice de estabilidad poblacional con cortes por deciles de la muestra esperada."""
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(cortes, esperado, side="right"), minlength=len(cortes) + 1)
    r = np.bincount(np.searchsorted(cortes, real, side="right"), minlength=len(cortes) + 1)
    e = np.clip(e / e.sum(), 1e-4, None)
    r = np.clip(r / r.sum(), 1e-4, None)
    return float(np.sum((r - e) * np.log(r / e)))


# ------------------------------------------------- política histórica SIMULADA

def regla_historica(df: pd.DataFrame, rng: np.random.Generator, fuga: float = 0.0) -> np.ndarray:
    """Puntaje de riesgo de la política histórica SIMULADA. Esta regla no existe en
    los datos: es un mecanismo inventado para este experimento.

    Suma puntos por señales que un analista revisaría a simple vista (chequera
    negativa, plazo largo, monto alto, ahorros bajos, antigüedad laboral corta) y
    agrega ruido que representa el juicio del analista. Con fuga > 0 el ruido se
    desplaza según el resultado real, como si el analista intuyera algo que no
    quedó en las variables (rechazo no ignorable)."""
    pts = (2.0 * (df["chequera"] == "A11") + 1.0 * (df["chequera"] == "A12")
           + 2.0 * (df["duracion"] > 24) + 1.0 * (df["monto"] > 5000)
           + 1.0 * (df["ahorros"] == "A61") + 1.0 * df["empleo"].isin(["A71", "A72"]))
    ruido = rng.normal(0.0, 1.0, len(df))
    return pts.to_numpy(float) + ruido + fuga * (df[OBJETIVO].to_numpy(float) - 0.3) / 0.46
