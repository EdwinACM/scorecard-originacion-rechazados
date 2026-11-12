"""Pruebas de integridad. Se corren con: python -m pytest"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from comun import (CANDIDATAS, OBJETIVO, SENSIBLES, Scorecard, WoE, cargar, dividir,
                   pd_a_puntos, psi, regla_historica)
from inferencia import entrenar_metodos, recuperacion, replica, simular_rechazo
import paso0_preparar_datos as p0


def test_base_completa_y_sin_nulos():
    df = cargar()
    assert len(df) == 1000
    assert df.isna().sum().sum() == 0
    assert set(df[OBJETIVO].unique()) == {0, 1}
    assert abs(df[OBJETIVO].mean() - 0.30) < 1e-9


def test_variables_sensibles_fuera_del_modelo():
    for v in SENSIBLES:
        assert v not in CANDIDATAS
    sc = Scorecard().fit(cargar(), cargar()[OBJETIVO])
    assert not set(sc.seleccion_) & set(SENSIBLES)


def test_suma_sha256_fijada_coincide_si_hay_zip():
    zip_ = p0.RAIZ / "datos" / "uci_german.zip"
    if zip_.exists():
        assert hashlib.sha256(zip_.read_bytes()).hexdigest() == p0.SHA256


def test_escala_de_puntos():
    # 50 buenos por cada malo = 600 puntos; duplicar los momios suma 20
    assert abs(float(pd_a_puntos(np.array([1 / 51]))[0]) - 600) < 0.5
    p = np.array([0.02, 0.10, 0.30, 0.60])
    assert (np.diff(pd_a_puntos(p)) < 0).all()
    momios = lambda x: (1 - x) / x
    assert abs((pd_a_puntos(np.array([0.1]))[0] - pd_a_puntos(np.array([0.2]))[0])
               - 20 * np.log2(momios(0.1) / momios(0.2))) < 1e-6


def test_woe_signo_e_iv():
    dev, _ = dividir(cargar())
    w = WoE(CANDIDATAS).fit(dev, dev[OBJETIVO])
    assert (w.iv_ >= 0).all()
    t = w.tablas_["chequera"].set_index("grupo")
    assert t.loc["A14", "woe"] > t.loc["A11", "woe"]          # sin chequera paga mejor
    assert t.loc["A14", "tasa_malos"] < t.loc["A11", "tasa_malos"]


def test_scorecard_puntos_bajan_con_pd():
    df = cargar()
    sc = Scorecard().fit(df, df[OBJETIVO])
    p, s = sc.pd_(df), sc.puntos(df)
    assert np.corrcoef(p, s)[0, 1] < -0.9
    tabla = sc.tabla_puntos()
    assert tabla["puntos"].notna().all()


def test_psi_identico_es_cero():
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) < 1e-9
    assert psi(x, x + 3) > 0.25


def test_rechazo_simulado_respeta_porcentaje_y_no_depende_del_resultado_sin_fuga():
    df = cargar()
    dev, prueba = dividir(df)
    rech, rech_pr = simular_rechazo(dev, prueba, 0.30, 0.0, np.random.default_rng(1))
    assert abs(rech.mean() - 0.30) < 0.01
    # sin fuga, el ruido no se mueve con el resultado
    a = regla_historica(dev, np.random.default_rng(5), 0.0)
    b = regla_historica(dev.assign(malo=1 - dev[OBJETIVO]), np.random.default_rng(5), 0.0)
    assert np.allclose(a, b)
    c = regla_historica(dev.assign(malo=1 - dev[OBJETIVO]), np.random.default_rng(5), 1.0)
    assert not np.allclose(a, c)


def test_los_metodos_de_inferencia_no_usan_el_resultado_de_rechazados():
    df = cargar()
    dev, prueba = dividir(df)
    rech, _ = simular_rechazo(dev, prueba, 0.30, 0.0, np.random.default_rng(1))
    alterado = dev.copy()
    alterado.loc[rech, OBJETIVO] = 1 - alterado.loc[rech, OBJETIVO]
    m1 = entrenar_metodos(dev, rech, np.random.default_rng(2))
    m2 = entrenar_metodos(alterado, rech, np.random.default_rng(2))
    for m in ["solo_aprobados", "augmentation", "parcelamiento", "fuzzy"]:
        assert np.allclose(m1[m].pd_(prueba), m2[m].pd_(prueba)), m
    assert not np.allclose(m1["oraculo"].pd_(prueba), m2["oraculo"].pd_(prueba))


def test_replica_y_reproducibilidad():
    df = cargar()
    a = replica(df, 42)
    b = replica(df, 42)
    assert len(a) == 5
    pd.testing.assert_frame_equal(a, b)
    assert a["auc"].between(0.5, 1).all()


def test_recuperacion():
    assert recuperacion(np.array([0.78]), np.array([0.75]), np.array([0.78])) == 1.0
    assert recuperacion(np.array([0.75]), np.array([0.75]), np.array([0.78])) == 0.0
