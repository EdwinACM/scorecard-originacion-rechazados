# Scorecard de originación e inferencia de rechazados

Proyecto personal en dos partes. La primera arma un scorecard clásico de originación con WoE, IV, regresión logística, escala de puntos, Gini, KS, PSI, curva de aprobación y corte. La segunda mide qué pasa cuando el modelo se entrena solo con los solicitantes aprobados y compara tres técnicas de inferencia de rechazados contra un oráculo que sí ve todos los resultados.

Es un ejercicio académico con datos públicos; no usa información de ningún banco ni de sus clientes.

## Los datos y lo que es simulado

Base Statlog German Credit de UCI (Hofmann, 1994): 1,000 solicitudes de crédito de un banco alemán, con 30% de malos. Tiene 20 variables y el resultado de cada solicitud.

La base no trae rechazados: todas las solicitudes tienen resultado. Por eso el mecanismo de rechazo de la segunda parte es simulado. Yo inventé una política histórica que puntúa a cada solicitante con una regla simple (chequera negativa, plazo mayor a 24 meses, monto mayor a 5,000, ahorros bajos, antigüedad laboral corta) más ruido que representa el juicio del analista, y rechaza al 30% con peor puntaje. A los rechazados se les oculta el resultado al entrenar. Esa política no existe en ningún banco. Sirve porque, al conocer el resultado real de todos, se puede medir el sesgo. Todo lo que dice "rechazados" en este repositorio se refiere a esa simulación.

## Qué encontré

### Scorecard clásico

Con división 70/30 estratificada (semilla 42), la regresión logística sobre variables en WoE con IV de al menos 0.02 conserva 12 de 17 variables. En la muestra de prueba (300 solicitudes) el AUC es 0.785, el Gini 0.569 y el KS 0.462; en desarrollo, AUC 0.816, así que hay algo de sobreajuste esperable con 700 filas. El PSI entre desarrollo y prueba es 0.012, aunque con 300 casos el PSI tiene un piso de ruido de varios centésimos, de modo que esa cifra dice poco. La variable más fuerte es el estado de la chequera (IV 0.63). La escala es 600 puntos con momios de 50 a 1 y 20 puntos por duplicar los momios. La tabla de puntos por grupo está en `resultados/01_scorecard_tabla.csv`.

### Corte propuesto

Con la matriz de costos de la documentación de la base (aprobar a un malo cuesta 5 veces lo que cuesta rechazar a un bueno) y puntajes de validación cruzada del desarrollo, el corte es 527 puntos. En prueba aprueba al 44.7% y los aprobados tienen 11.2% de malos, contra 30% en la población. Es un corte estricto por el costo asimétrico; con otra matriz de costos cambiaría.

### Variables sensibles

Estado civil y sexo (vienen juntos en una columna), edad y condición de trabajador extranjero quedan fuera del modelo. Agregar estado civil/sexo y extranjero sube el AUC de prueba de 0.785 a 0.787, una diferencia que no distingo del ruido, así que excluirlas casi no cuesta poder predictivo. No audité proxies (por ejemplo antigüedad laboral o vivienda pueden correlacionar con la edad) ni medí impacto desigual entre grupos.

### Sesgo por truncamiento

Con el rechazo del 30% y sin fuga de información, entrenar solo con aprobados baja el AUC en la población completa de prueba de 0.781 a 0.755 en promedio de 300 repeticiones, una pérdida de 0.026 (intervalo de 95%: 0.023 a 0.028). El KS cae 0.053. El modelo sesgado también subestima la tasa de malos en 2.9 puntos. En la zona que la política habría rechazado, el AUC baja de 0.701 a 0.670.

### Qué recupera cada método

Promedio de 300 repeticiones. La cifra es la fracción de la brecha entre el modelo sesgado y el oráculo que cierra el método; 1 es igualar al oráculo.

- Fuzzy augmentation: 0.91 en AUC (0.86 a 0.96).
- Parcelamiento: 0.78 en AUC (0.73 a 0.82).
- Augmentation por reponderación: negativa, -0.07 (-0.10 a -0.05). No ayudó y empeoró un poco.

Ambos métodos que reclasifican suben el AUC unos 0.02 sobre el modelo sesgado y mejoran en 91% y 96% de las repeticiones (`resultados/02_parteB_pareadas_vs_sesgado.csv`). El costo es la calibración: con el supuesto de que los rechazados tienen el doble de malos que predice el modelo de aprobados, ambos sobreestiman la PD en 5.2 puntos. En esta simulación el supuesto está mal, porque el multiplicador real es cercano a 1 (en la corrida de semilla 42 los rechazados tienen 46.7% de malos y el modelo de aprobados predecía 46.0%). Con factor 1 el parcelamiento y el fuzzy dejan de sobreestimar pero recuperan menos AUC (0.756 y 0.762 contra 0.752 sesgado y 0.775 oráculo); con factor 3 el AUC se acerca al oráculo y la PD sobreestimada llega a 8.4 puntos (`resultados/03_sensibilidad_factor.csv`).

### Una sola corrida no alcanza para ver el efecto

En la división de semilla 42, con bootstrap sobre los 300 casos de prueba, el AUC es 0.785 (oráculo), 0.760 (solo aprobados), 0.744 (augmentation), 0.767 (parcelamiento) y 0.772 (fuzzy). Los intervalos miden unos 0.11 de ancho y se traslapan por completo; la diferencia pareada de fuzzy contra sesgado es 0.012 con intervalo de -0.010 a 0.034. Con una muestra de prueba así de chica, el efecto de 0.02 a 0.03 de AUC no se detecta en una corrida.

### Sensibilidad

Sin fuga, el sesgo crece con el porcentaje rechazado: -0.001 de AUC al rechazar 10%, -0.023 al 30% y -0.052 al 50%. Cuando el analista "intuye" el resultado y la regla de rechazo lo refleja (fuga 0.5 y 1.0), el sesgo es mucho mayor (-0.065 y -0.131 al 30%) y los métodos recuperan una parte, no todo: con fuga 1.0 y 30% de rechazo el AUC sube de 0.645 a 0.699 con fuzzy, lejos de 0.775 del oráculo. Ahí ningún método basado en las variables observadas puede reconstruir lo que no está en ellas. Los escenarios están en `resultados/03_sensibilidad_rechazo_fuga.csv` y `03_sesgo_por_escenario.csv`.

## Límites

- El mecanismo de rechazo es simulado. Los resultados dependen de la regla que inventé, del ruido y del parámetro de fuga; en un banco real el mecanismo es desconocido y puede no parecerse.
- Son 1,000 observaciones. En la prueba de 300 casos los intervalos son anchos. Los intervalos de las 300 repeticiones son angostos porque cada repetición re-divide las mismas 1,000 solicitudes: miden la variación del proceso de simulación y no la incertidumbre sobre una población nueva, que es mayor.
- El factor de inflación de malos (2) es un supuesto fijo, no se ajustó con los resultados. Se probó 1, 1.5, 2 y 3.
- El corte se calculó con puntajes de validación cruzada y costos de la documentación de la base; no son costos de una cartera real. La base es alemana y antigua, con montos en marcos.
- La agrupación del WoE (cuantiles, categorías pequeñas juntas) y la selección por IV se reajustan en cada repetición, pero no revisé monotonía ni estabilidad por variable.
- La variable de edad no se probó en el modelo de referencia con sensibles.

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/paso0_preparar_datos.py   # opcional: el CSV ya viene en datos/
python src/ejecutar_todo.py          # unos 2.5 minutos
python -m pytest
```

El paso 0 descarga el zip de UCI, verifica su SHA-256 y extrae solo `german.data`. Los pasos: `paso1_scorecard.py`, `paso2_rechazados.py` y `paso3_sensibilidad.py`; el código del experimento está en `src/inferencia.py` y el del scorecard en `src/comun.py`. Los resultados quedan en `resultados/` como CSV, JSON y tres gráficas. La semilla es 42 en todos los pasos. Una acción de GitHub corre las pruebas en cada cambio.

`docs/bitacora.md` registra las decisiones del análisis.

## Referencia

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.

Licencia MIT.
