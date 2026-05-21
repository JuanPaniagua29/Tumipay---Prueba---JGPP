-- ============================================================
-- TUMIPAY – Prueba Técnica Data Science Engineer
-- Consultas SQL para transformación y análisis
-- Dialect: PostgreSQL
-- Fecha de corte: 2026-04-30
-- ============================================================


-- ============================================================
-- 1. CARGA DE TABLAS BASE (usa schema_postgresql.sql primero)
-- ============================================================
-- Ver archivo schema_postgresql.sql para la creación de tablas.
-- Los CSV se cargan con: \COPY tabla FROM 'archivo.csv' CSV HEADER;


-- ============================================================
-- 2. KPIs GENERALES DEL PORTAFOLIO
-- ============================================================
SELECT
    COUNT(DISTINCT c.cliente_id)                        AS total_clientes,
    COUNT(DISTINCT cr.credito_id)                       AS total_creditos,
    SUM(cr.monto_credito)                               AS monto_total_desembolsado,
    AVG(cr.monto_credito)                               AS monto_promedio,
    AVG(cr.plazo_meses)                                 AS plazo_promedio_meses,
    AVG(cr.tasa_interes_mensual)                        AS tasa_promedio_mensual,
    COUNT(DISTINCT CASE WHEN cr.estado_credito_operativo IN ('Mora moderada','Mora severa')
                        THEN cr.credito_id END)         AS creditos_en_mora,
    ROUND(
        COUNT(DISTINCT CASE WHEN cr.estado_credito_operativo IN ('Mora moderada','Mora severa')
                            THEN cr.credito_id END)::NUMERIC
        / COUNT(DISTINCT cr.credito_id) * 100, 2
    )                                                   AS pct_en_mora
FROM clientes c
LEFT JOIN creditos cr ON c.cliente_id = cr.cliente_id;


-- ============================================================
-- 3. CONSTRUCCIÓN DE VARIABLE OBJETIVO: mora_30
--    Crédito con al menos 1 cuota (primeras 6) con días_mora > 30
-- ============================================================
WITH primeras_cuotas AS (
    SELECT
        credito_id,
        MAX(dias_mora)                                      AS max_dias_mora_6m,
        COUNT(*)                                            AS cuotas_observadas,
        SUM(CASE WHEN estado_pago = 'Pagado' THEN 1 ELSE 0 END) AS cuotas_pagadas_a_tiempo
    FROM pagos
    WHERE numero_cuota <= 6
    GROUP BY credito_id
    HAVING COUNT(*) >= 3   -- mínimo 3 cuotas observadas
)
SELECT
    credito_id,
    max_dias_mora_6m,
    cuotas_observadas,
    ROUND(cuotas_pagadas_a_tiempo::NUMERIC / cuotas_observadas, 4) AS pct_pagadas_a_tiempo,
    CASE WHEN max_dias_mora_6m > 30 THEN 1 ELSE 0 END              AS mora_30
FROM primeras_cuotas;


-- ============================================================
-- 4. TABLA ANALÍTICA BASE PARA MODELADO
--    Une clientes + créditos + target de mora
-- ============================================================
WITH target AS (
    SELECT
        credito_id,
        MAX(dias_mora)                                       AS max_dias_mora_6m,
        CASE WHEN MAX(dias_mora) > 30 THEN 1 ELSE 0 END     AS mora_30,
        ROUND(
            SUM(CASE WHEN estado_pago = 'Pagado' THEN 1 ELSE 0 END)::NUMERIC / COUNT(*), 4
        )                                                    AS pct_pagadas_a_tiempo
    FROM pagos
    WHERE numero_cuota <= 6
    GROUP BY credito_id
    HAVING COUNT(*) >= 3
),
resumen_pagos AS (
    SELECT
        p.credito_id,
        COUNT(*)                                             AS total_cuotas,
        SUM(p.valor_pagado)                                  AS total_pagado,
        SUM(p.valor_cuota)                                   AS total_esperado,
        MAX(p.dias_mora)                                     AS max_dias_mora_total,
        SUM(CASE WHEN p.dias_mora > 0 THEN 1 ELSE 0 END)    AS cuotas_con_mora
    FROM pagos p
    GROUP BY p.credito_id
)
SELECT
    cl.cliente_id,
    cl.edad,
    cl.genero,
    cl.estrato,
    cl.nivel_educativo,
    cl.ocupacion,
    cl.ingreso_mensual_estimado,
    cl.canal_adquisicion,
    cl.score_externo,
    cl.tiene_producto_ahorro,
    cl.numero_dependientes,
    cr.credito_id,
    cr.fecha_desembolso,
    cr.producto_credito,
    cr.monto_credito,
    cr.plazo_meses,
    cr.tasa_interes_mensual,
    cr.valor_cuota_pactada,
    cr.canal_originacion,
    cr.score_interno_originacion,
    cr.relacion_cuota_ingreso,
    cr.politica_aprobacion,
    cr.estado_credito_operativo,
    rp.total_cuotas,
    rp.total_pagado,
    rp.cuotas_con_mora,
    t.max_dias_mora_6m,
    t.pct_pagadas_a_tiempo,
    t.mora_30
FROM clientes cl
JOIN creditos cr      ON cl.cliente_id = cr.cliente_id
JOIN target   t       ON cr.credito_id = t.credito_id
JOIN resumen_pagos rp ON cr.credito_id = rp.credito_id;


-- ============================================================
-- 5. TASA DE MORA POR CANAL DE ORIGINACIÓN
-- ============================================================
WITH target AS (
    SELECT credito_id,
           CASE WHEN MAX(dias_mora) > 30 THEN 1 ELSE 0 END AS mora_30
    FROM pagos WHERE numero_cuota <= 6
    GROUP BY credito_id HAVING COUNT(*) >= 3
)
SELECT
    cr.canal_originacion,
    COUNT(*)                                AS n_creditos,
    SUM(t.mora_30)                          AS en_mora,
    ROUND(AVG(t.mora_30) * 100, 2)          AS pct_mora,
    ROUND(AVG(cr.monto_credito), 0)         AS monto_promedio,
    ROUND(AVG(cr.score_interno_originacion), 1) AS score_promedio
FROM creditos cr
JOIN target t ON cr.credito_id = t.credito_id
GROUP BY cr.canal_originacion
ORDER BY pct_mora DESC;


-- ============================================================
-- 6. TASA DE MORA POR PRODUCTO Y ESTRATO
-- ============================================================
WITH target AS (
    SELECT credito_id,
           CASE WHEN MAX(dias_mora) > 30 THEN 1 ELSE 0 END AS mora_30
    FROM pagos WHERE numero_cuota <= 6
    GROUP BY credito_id HAVING COUNT(*) >= 3
)
SELECT
    cr.producto_credito,
    cl.estrato,
    COUNT(*)                              AS n_creditos,
    ROUND(AVG(t.mora_30) * 100, 2)        AS pct_mora,
    ROUND(AVG(cr.monto_credito), 0)       AS monto_promedio,
    ROUND(AVG(cl.ingreso_mensual_estimado), 0) AS ingreso_promedio
FROM creditos cr
JOIN clientes cl ON cr.cliente_id = cl.cliente_id
JOIN target   t  ON cr.credito_id = t.credito_id
WHERE cl.estrato IS NOT NULL
GROUP BY cr.producto_credito, cl.estrato
ORDER BY cr.producto_credito, cl.estrato;


-- ============================================================
-- 7. COMPORTAMIENTO DE PAGO: CUOTAS VENCIDAS Y MORA ACUMULADA
--    Función ventana para ver evolución por crédito
-- ============================================================
SELECT
    p.credito_id,
    p.numero_cuota,
    p.fecha_vencimiento,
    p.fecha_pago,
    p.dias_mora,
    p.estado_pago,
    p.valor_cuota,
    p.valor_pagado,
    -- acumulado de días de mora hasta esta cuota
    SUM(p.dias_mora) OVER (
        PARTITION BY p.credito_id
        ORDER BY p.numero_cuota
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS mora_acumulada,
    -- máximo días mora hasta esta cuota
    MAX(p.dias_mora) OVER (
        PARTITION BY p.credito_id
        ORDER BY p.numero_cuota
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS max_mora_hasta_cuota,
    -- número de cuotas en mora hasta esta cuota
    COUNT(CASE WHEN p.dias_mora > 0 THEN 1 END) OVER (
        PARTITION BY p.credito_id
        ORDER BY p.numero_cuota
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cuotas_con_mora_acumuladas
FROM pagos p
ORDER BY p.credito_id, p.numero_cuota;


-- ============================================================
-- 8. CLIENTES DE ALTO RIESGO: PERFIL CONSOLIDADO
--    Clientes con mora en cualquier crédito
-- ============================================================
WITH clientes_mora AS (
    SELECT DISTINCT p.cliente_id
    FROM pagos p
    WHERE p.dias_mora > 30
)
SELECT
    cl.cliente_id,
    cl.edad,
    cl.genero,
    cl.estrato,
    cl.ocupacion,
    cl.ingreso_mensual_estimado,
    cl.score_externo,
    COUNT(DISTINCT cr.credito_id)           AS num_creditos,
    SUM(cr.monto_credito)                   AS exposicion_total,
    MAX(p.dias_mora)                        AS max_dias_mora,
    COUNT(CASE WHEN p.dias_mora > 30 THEN 1 END) AS cuotas_en_mora_30
FROM clientes cl
JOIN clientes_mora cm ON cl.cliente_id = cm.cliente_id
JOIN creditos cr ON cl.cliente_id = cr.cliente_id
JOIN pagos    p  ON cr.credito_id  = p.credito_id
GROUP BY cl.cliente_id, cl.edad, cl.genero, cl.estrato,
         cl.ocupacion, cl.ingreso_mensual_estimado, cl.score_externo
ORDER BY max_dias_mora DESC, exposicion_total DESC;


-- ============================================================
-- 9. KPIs PARA DASHBOARD POWER BI
--    Resumen mensual de desembolsos y mora
-- ============================================================
SELECT
    DATE_TRUNC('month', cr.fecha_desembolso)            AS mes_desembolso,
    COUNT(*)                                            AS creditos_desembolsados,
    SUM(cr.monto_credito)                               AS monto_desembolsado,
    AVG(cr.score_interno_originacion)                   AS score_promedio,
    SUM(CASE WHEN cr.estado_credito_operativo IN ('Mora moderada','Mora severa')
             THEN 1 ELSE 0 END)                         AS en_mora,
    ROUND(
        SUM(CASE WHEN cr.estado_credito_operativo IN ('Mora moderada','Mora severa')
                 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100, 2
    )                                                   AS pct_mora_operativa
FROM creditos cr
GROUP BY DATE_TRUNC('month', cr.fecha_desembolso)
ORDER BY mes_desembolso;


-- ============================================================
-- 10. SEGMENTACIÓN RFM ADAPTADA (Recency-Frequency-Mora)
--     Para Operaciones: clientes según comportamiento de pago
-- ============================================================
WITH comportamiento AS (
    SELECT
        p.cliente_id,
        MAX(p.fecha_pago)                                      AS ultimo_pago,
        COUNT(DISTINCT p.credito_id)                           AS num_creditos,
        AVG(p.dias_mora)                                       AS promedio_dias_mora,
        MAX(p.dias_mora)                                       AS max_dias_mora,
        SUM(CASE WHEN p.estado_pago = 'Pagado' THEN 1 ELSE 0 END)::NUMERIC
            / NULLIF(COUNT(*), 0)                              AS pct_pagos_puntuales
    FROM pagos p
    WHERE p.fecha_pago IS NOT NULL
    GROUP BY p.cliente_id
)
SELECT
    c.cliente_id,
    c.ultimo_pago,
    c.num_creditos,
    ROUND(c.promedio_dias_mora, 1)     AS promedio_dias_mora,
    c.max_dias_mora,
    ROUND(c.pct_pagos_puntuales * 100, 1) AS pct_puntuales,
    CASE
        WHEN c.max_dias_mora = 0              THEN 'Excelente'
        WHEN c.max_dias_mora BETWEEN 1 AND 15 THEN 'Bueno'
        WHEN c.max_dias_mora BETWEEN 16 AND 30 THEN 'Regular'
        WHEN c.max_dias_mora BETWEEN 31 AND 60 THEN 'En riesgo'
        ELSE 'Crítico'
    END AS segmento_comportamiento
FROM comportamiento c
ORDER BY c.max_dias_mora DESC;
