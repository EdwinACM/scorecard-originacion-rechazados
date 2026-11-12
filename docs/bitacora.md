# Bitácora de decisiones

Fecha de la corrida: 12 de noviembre de 2026. Semilla 42 en todos los pasos.

## Datos

1. Se eligió Statlog German Credit de UCI porque es gratuita, no pide cuenta y trae el resultado de cada solicitud. La suma SHA-256 del zip se calculó en la primera descarga y quedó fijada en el paso 0.
2. Se usa `german.data` (variables categóricas originales) y se extrae solo ese archivo. En la base original la clase 2 es malo; se recodificó a malo = 1.
3. Estado civil y sexo vienen en una sola columna, así que se excluyen juntos. También se excluyen edad y trabajador extranjero.

## Scorecard

4. Las variables continuas (plazo y monto) se agrupan en cuantiles de cinco partes; las demás usan sus categorías y las que tienen menos de 5% de los casos se juntan en "otras". El WoE usa suavizado de 0.5.
5. Entran las variables con IV de al menos 0.02. Con eso quedan 12 de 17.
6. Regresión logística con regularización L2 por defecto (C = 1). El intercepto se reparte en partes iguales entre las variables al calcular puntos.
7. Escala: 600 puntos con momios de 50 a 1 y 20 puntos por duplicar. Factor 28.85 y offset 487.12.
8. El corte minimiza el costo con 5 por malo aprobado y 1 por bueno rechazado, sobre puntajes de validación cruzada del desarrollo.
9. Se comparó el modelo con y sin estado civil/sexo y extranjero para documentar el costo de excluirlos (AUC de prueba 0.787 contra 0.785).

## Rechazados

10. Política histórica simulada: puntos por chequera, plazo, monto, ahorros y antigüedad, más ruido normal. Rechaza al 30% con peor puntaje en desarrollo. En prueba se marca quién habría sido rechazado con el mismo corte, para evaluar en la zona que el sesgado nunca vio.
11. El parámetro de fuga mueve el puntaje según el resultado real, para representar rechazo no ignorable. En el escenario principal es 0.
12. Métodos: oráculo, solo aprobados, augmentation (peso por banda de score del modelo de aprobados, 10 bandas), parcelamiento (5 parcelas entre rechazados, etiquetas al azar con la tasa inflada) y fuzzy (cada rechazado entra como malo con peso p y como bueno con peso 1-p).
13. El factor de inflación es 2, un supuesto declarado y no ajustado a los resultados. La prueba `test_los_metodos_de_inferencia_no_usan_el_resultado_de_rechazados` comprueba que ningún método salvo el oráculo ve el resultado de los rechazados.
14. Dos formas de intervalo: bootstrap de 2,000 sobre la muestra de prueba en la corrida de semilla 42, y 300 repeticiones completas con bootstrap de 2,000 sobre las repeticiones. La segunda es angosta porque las repeticiones comparten las mismas 1,000 solicitudes.

## Cambios durante el análisis

15. Una primera versión del WoE categórico tenía una rama confusa para el grupo "otras". Se reescribió: las categorías no vistas al ajustar caen en un grupo extra con WoE 0.
16. Se descubrió al revisar resultados que el factor 2 es demasiado alto para este mecanismo (el multiplicador real es cercano a 1), lo que sobreestima la PD en los métodos de reclasificación. Se dejó el supuesto original y se agregó el análisis de sensibilidad al factor en lugar de cambiarlo a posteriori.
