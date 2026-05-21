# TUMIPAY · Prueba Técnica Data Science Engineer

> Análisis end-to-end de riesgo de mora en créditos fintech: EDA, SQL, modelado predictivo y visualización.

---

## Resumen ejecutivo

Se analizó un portafolio de **1,411 créditos** con fecha de corte 2026-04-30.  
La tasa de mora definida (mora_30) es del **23.95%**.  
El modelo de Random Forest alcanza un **AUC-ROC de 0.68 en test** y **0.70 ± 0.04 en validación cruzada**.  
Las bandas de riesgo predichas muestran separación monótona: de **2.7% (Bajo)** hasta **75% (Muy alto)**.

**Hallazgo principal:** El score interno, la relación cuota/ingreso y el score externo son los tres factores con mayor poder predictivo al momento del desembolso.

---

## Estructura del repositorio

```
tumipay/
├── data/
│   ├── clientes.csv                  # Maestro de clientes (original)
│   ├── creditos.csv                  # Créditos desembolsados (original)
│   ├── pagos.csv                     # Historial de cuotas (original)
│   ├── eventos_app.csv               # Eventos digitales (original)
│   ├── dataset_analitico.csv         # GENERADO: créditos + clientes + eventos + target
│   ├── dataset_con_scores.csv        # GENERADO: incluye prob_mora y banda_riesgo
│   ├── mora_target.csv               # GENERADO: variable objetivo por crédito
│   └── features_eventos.csv          # GENERADO: features de comportamiento digital
│
├── notebooks/
│   ├── 01_EDA_calidad_datos.ipynb    # Exploración, calidad, target, distribuciones
│   ├── 02_insights_negocio.ipynb     # 4 preguntas de negocio con evidencia
│   └── 03_modelado_ML.ipynb          # Features, modelos, evaluación, bandas de riesgo
│
├── sql/
│   └── consultas_analiticas.sql      # 10 consultas PostgreSQL documentadas
│
├── dashboard/                        # Gráficos exportados (14 PNG)
│   ├── g01_distribuciones.png
│   ├── g02_calidad_datos.png
│   ├── g03_target.png
│   ├── g04_mora_canal_producto.png
│   ├── g05_score_rci_vs_mora.png
│   ├── g06_mora_demografica.png
│   ├── g07_evolucion_pago_cuota.png
│   ├── g08_eventos_vs_mora.png
│   ├── g09_correlaciones.png
│   ├── g10_roc_pr.png
│   ├── g11_importancia_variables.png
│   ├── g12_bandas_riesgo.png
│   ├── g13_insights_negocio.png
│   └── g14_operativo.png
│
├── requirements.txt
└── README.md
```

---

## Instrucciones de ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar en orden
jupyter notebook notebooks/01_EDA_calidad_datos.ipynb
jupyter notebook notebooks/02_insights_negocio.ipynb
jupyter notebook notebooks/03_modelado_ML.ipynb
```

Los notebooks 02 y 03 dependen de los archivos generados por el 01.  
El SQL en `/sql/consultas_analiticas.sql` requiere PostgreSQL con los CSV cargados usando `schema_postgresql.sql`.

---

## Variable objetivo — definición y justificación

**`mora_30 = 1`** si el crédito registra al menos una cuota en las **primeras 6** con `dias_mora > 30`.

| Decisión | Justificación |
|---|---|
| Umbral 30 días | Estándar regulatorio colombiano para mora C (Superfinanciera) |
| Primeras 6 cuotas | Evita fuga de información: solo usamos señal temprana disponible en originación |
| Mínimo 3 cuotas observadas | Garantiza señal suficiente para etiquetar el crédito |
| Deduplicación previa | Cuotas duplicadas resueltas conservando mayor valor_pagado |

---

## Hallazgos principales

### Factores asociados a mayor mora
1. **Score interno bajo** — diferencia de 38 puntos promedio entre grupos (827 vs 789). Es el predictor más potente.
2. **Relación cuota/ingreso alta** — clientes con RCI > 0.35 tienen mora del 36% vs 18% en los demás (+49% mayor).
3. **Ingreso mensual bajo** — diferencia del 12.7% entre grupos.
4. **Score externo bajo** — diferencia del 9.1%.
5. **Canal Aliado + Microcrédito** — segmento con mayor tasa de mora: 39.7% (n=63).

### Segmentos de mayor riesgo
- **Canal:** Aliado (mayor tasa de mora)
- **Producto:** Microcrédito y Crédito libre inversión
- **Estrato:** 1 y 2
- **Ocupación:** Desempleado e Independiente

### Patrones de comportamiento de pago
- La mora se manifiesta desde la **cuota 1** (no hay período de gracia natural)
- El porcentaje de pago puntual es estable (~78-80%) sin deterioro progresivo claro
- Los pagos parciales son bajos y constantes (~2%)
- Clientes con mora registran más pagos fallidos en app (señal digital de alerta)

### Recomendaciones accionables

**Riesgo:**
- Revisar umbral de score interno en originación (< 790 → mora 30%+)
- Aplicar límite duro de RCI ≤ 0.35
- Implementar el modelo de bandas como capa adicional de aprobación

**Producto:**
- Auditar controles de originación en canal Aliado
- Revisar condiciones del Microcrédito (plazos, cuotas)

**Operaciones:**
- Activar cobranza preventiva antes del vencimiento de cuota 1
- Usar pagos_fallidos en app como señal de alerta temprana

---

## Modelo de Machine Learning

| Modelo | AUC Test | AUC CV (5-fold) |
|---|---|---|
| Regresión Logística | 0.7064 | 0.7015 ± 0.044 |
| Random Forest | 0.6847 | 0.6972 ± 0.042 |

**Modelo seleccionado:** Regresión Logística — desempeño ligeramente superior, mayor interpretabilidad, apto para auditoría regulatoria.

**Top 5 variables:** score_externo, score_interno_originacion, ingreso_mensual_estimado, relacion_cuota_ingreso, edad.

**Bandas de riesgo:**

| Banda | N | Tasa mora real |
|---|---|---|
| Bajo (< 15%) | 73 | 2.7% |
| Medio (15-35%) | 484 | 3.5% |
| Alto (35-60%) | 642 | 24.9% |
| Muy alto (> 60%) | 212 | 75.0% |

---

## Dashboard Power BI — descripción de páginas

*(Gráficos en /dashboard/. Archivo .pbix disponible bajo solicitud.)*

**Página 1 — Resumen ejecutivo:**  
KPIs globales (créditos, monto, tasa mora, score promedio), evolución mensual de desembolsos y mora, estados operativos. *Decisión que permite: monitorear el portafolio a nivel gerencial.*

**Página 2 — Segmentación y riesgo:**  
Mapa de calor canal × producto, mora por estrato, distribución de scores, bandas de riesgo del modelo. *Decisión: identificar segmentos prioritarios para ajuste de política.*

**Página 3 — Comportamiento de pago:**  
Evolución mora por cuota, puntualidad por segmento, comparación comportamiento digital (pagos fallidos, tasa de éxito). *Decisión: priorizar acciones de cobranza temprana y alertas digitales.*

---

## Supuestos documentados

1. La variable objetivo se construye con las primeras 6 cuotas para anticipar riesgo sin usar información futura.
2. Créditos con < 3 cuotas observadas se excluyen del modelado por información insuficiente.
3. La imputación de ingreso por mediana de ocupación asume que la ocupación predice mejor el ingreso que la media global.
4. Los eventos_app se tratan como comportamiento previo/contemporáneo al desembolso.
5. Los email_hash duplicados se reportan como hallazgo pero no se eliminan sin validación de negocio.
6. El dataset es sintético; los patrones encontrados requieren validación en datos de producción.

---

## Limitaciones y mejoras propuestas

- **Validación temporal:** En producción, separar test por fecha (out-of-time), no aleatoriamente.
- **XGBoost/LightGBM:** Típicamente superan RF en datos tabulares de crédito.
- **SHAP values:** Para explicabilidad individual auditable por decisión de crédito.
- **Calibración de probabilidades:** Para que el score sea estable y comparable en el tiempo.
- **Análisis de fairness:** Revisar si el modelo genera disparidades por género o estrato antes de desplegar.
- **PSI (Population Stability Index):** Para detectar deriva del modelo en producción.

---

## Uso de IA generativa

Se utilizó **Claude (Anthropic)** como asistente para:
- Estructurar el proyecto y definir la metodología paso a paso
- Generar borradores de código Python y SQL revisados y adaptados
- Redactar secciones de documentación y README

Todo el contenido fue revisado, entendido y puede ser defendido en la sustentación técnica.
