"""
feature_engineering.py
======================
TUMIPAY — Prueba Técnica Data Science Engineer

Funciones reutilizables para construir el dataset analítico desde los CSVs crudos.
Este módulo centraliza todas las transformaciones para que sean reproducibles
tanto en notebooks como en un pipeline de producción.

Uso:
    from src.feature_engineering import build_dataset
    df = build_dataset(data_path='data/')
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONSTANTES DE NEGOCIO
# ============================================================
MORA_UMBRAL_DIAS    = 30   # días de mora para clasificar como mora_30
CUOTAS_HORIZONTE    = 6    # primeras N cuotas para construir el target
CUOTAS_MIN          = 3    # mínimo de cuotas observadas para etiquetar
FECHA_CORTE         = pd.Timestamp('2026-04-30')
RCI_MAX_VALIDO      = 5.0  # relacion_cuota_ingreso mayor a esto = error de datos
TASA_OUTLIER_UMBRAL = 0.10 # tasa mensual > 10% se marca como outlier
EDAD_MIN            = 18
EDAD_MAX            = 80


# ============================================================
# 1. CARGA DE DATOS
# ============================================================
def load_raw_data(data_path: str = 'data/') -> dict:
    """
    Carga los 4 CSVs crudos y parsea fechas.

    Returns:
        dict con keys: 'clientes', 'creditos', 'pagos', 'eventos'
    """
    path = Path(data_path)
    print(f"[load] Leyendo datos desde {path.resolve()}")

    clientes = pd.read_csv(path / 'clientes.csv')
    creditos = pd.read_csv(path / 'creditos.csv')
    pagos    = pd.read_csv(path / 'pagos.csv')
    eventos  = pd.read_csv(path / 'eventos_app.csv')

    # Parsear fechas
    creditos['fecha_desembolso']    = pd.to_datetime(creditos['fecha_desembolso'])
    pagos['fecha_vencimiento']      = pd.to_datetime(pagos['fecha_vencimiento'])
    pagos['fecha_pago']             = pd.to_datetime(pagos['fecha_pago'])
    eventos['fecha_evento']         = pd.to_datetime(eventos['fecha_evento'])

    print(f"  clientes : {clientes.shape[0]:,} filas")
    print(f"  creditos : {creditos.shape[0]:,} filas")
    print(f"  pagos    : {pagos.shape[0]:,} filas")
    print(f"  eventos  : {eventos.shape[0]:,} filas")

    return {'clientes': clientes, 'creditos': creditos,
            'pagos': pagos, 'eventos': eventos}


# ============================================================
# 2. LIMPIEZA Y NORMALIZACIÓN
# ============================================================
def clean_clientes(clientes: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza la tabla de clientes.

    Decisiones:
    - Normaliza errores tipográficos en ciudad
    - Imputa ingreso_mensual_estimado con mediana por ocupacion
    - Imputa score_externo con mediana global + crea flag de nulo
    - Marca edades fuera de rango como sospechosas
    """
    df = clientes.copy()

    # Normalizar ciudad
    ciudad_corrections = {
        'Bogot':       'Bogotá',
        'Barranquila': 'Barranquilla',
        'Medellin ':   'Medellín',
    }
    df['ciudad'] = df['ciudad'].replace(ciudad_corrections)

    # Imputar ingreso por mediana de ocupacion
    df['ingreso_mensual_estimado'] = df.groupby('ocupacion')['ingreso_mensual_estimado']\
        .transform(lambda x: x.fillna(x.median()))
    # Si sigue nulo (ocupacion sin ningún valor), usar mediana global
    df['ingreso_mensual_estimado'] = df['ingreso_mensual_estimado']\
        .fillna(df['ingreso_mensual_estimado'].median())

    # Score externo: flag + imputar
    df['score_externo_nulo'] = df['score_externo'].isna().astype(int)
    df['score_externo']      = df['score_externo'].fillna(df['score_externo'].median())

    # Edad sospechosa
    df['edad_sospechosa'] = (
        (df['edad'] < EDAD_MIN) | (df['edad'] > EDAD_MAX)
    ).astype(int)

    return df


def clean_creditos(creditos: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza la tabla de créditos.

    Decisiones:
    - Imputa producto_credito nulo con 'Desconocido'
    - Corrige relacion_cuota_ingreso negativa con mediana
    - Marca tasas de interés outlier
    """
    df = creditos.copy()

    # Producto nulo
    df['producto_credito'] = df['producto_credito'].fillna('Desconocido')

    # RCI negativo = error de datos
    mediana_rci = df.loc[df['relacion_cuota_ingreso'] >= 0, 'relacion_cuota_ingreso'].median()
    df.loc[df['relacion_cuota_ingreso'] < 0, 'relacion_cuota_ingreso'] = mediana_rci

    # Tasa outlier
    df['tasa_outlier'] = (df['tasa_interes_mensual'] > TASA_OUTLIER_UMBRAL).astype(int)

    return df


def clean_pagos(pagos: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la tabla de pagos.

    Decisiones:
    - Elimina duplicados de (credito_id, numero_cuota) conservando
      el registro con mayor valor_pagado (más representativo)
    """
    df = pagos.copy()
    n_antes = len(df)
    df = df.sort_values('valor_pagado', ascending=False)\
           .drop_duplicates(['credito_id', 'numero_cuota'], keep='first')
    n_despues = len(df)
    if n_antes > n_despues:
        print(f"  [clean_pagos] Eliminados {n_antes - n_despues} duplicados de cuota")
    return df


# ============================================================
# 3. CONSTRUCCIÓN DE VARIABLE OBJETIVO
# ============================================================
def build_target(pagos_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la variable objetivo mora_30 a nivel de crédito.

    Definición: mora_30 = 1 si el crédito tiene al menos una cuota
    en las primeras CUOTAS_HORIZONTE con dias_mora > MORA_UMBRAL_DIAS.

    Justificación:
    - 30 días: umbral regulatorio colombiano para mora C (Superfinanciera)
    - Primeras 6 cuotas: evita fuga de información
    - Mínimo 3 cuotas: garantiza señal suficiente

    Returns:
        DataFrame con columnas: credito_id, mora_30, max_dias_mora_6m,
        pct_pagadas_tiempo, n_cuotas_con_mora, cuotas_observadas
    """
    primeras = pagos_clean[pagos_clean['numero_cuota'] <= CUOTAS_HORIZONTE]

    target = primeras.groupby('credito_id').agg(
        max_dias_mora_6m   = ('dias_mora',    'max'),
        cuotas_observadas  = ('numero_cuota', 'count'),
        pct_pagadas_tiempo = ('estado_pago',  lambda x: (x == 'Pagado').mean()),
        n_cuotas_con_mora  = ('dias_mora',    lambda x: (x > 0).sum()),
    ).reset_index()

    # Filtrar créditos con suficiente información
    target = target[target['cuotas_observadas'] >= CUOTAS_MIN]
    target['mora_30'] = (target['max_dias_mora_6m'] > MORA_UMBRAL_DIAS).astype(int)

    tasa = target['mora_30'].mean()
    print(f"  [build_target] {len(target):,} créditos etiquetados | Tasa mora: {tasa:.2%}")

    return target


# ============================================================
# 4. FEATURES DE EVENTOS DIGITALES
# ============================================================
def build_eventos_features(eventos: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features de comportamiento digital por cliente.

    Solo usa eventos hasta la FECHA_CORTE para evitar fuga de información.

    Returns:
        DataFrame con una fila por cliente y features de comportamiento digital.
    """
    ev = eventos[eventos['fecha_evento'] <= FECHA_CORTE].copy()

    feats = ev.groupby('cliente_id').agg(
        total_eventos        = ('evento_id',          'count'),
        dias_activo          = ('fecha_evento',        lambda x: (x.max() - x.min()).days + 1),
        logins               = ('tipo_evento',         lambda x: (x == 'login').sum()),
        pagos_iniciados      = ('tipo_evento',         lambda x: (x == 'pago_iniciado').sum()),
        pagos_exitosos       = ('tipo_evento',         lambda x: (x == 'pago_exitoso').sum()),
        pagos_fallidos       = ('tipo_evento',         lambda x: (x == 'pago_fallido').sum()),
        consultas_saldo      = ('tipo_evento',         lambda x: (x == 'consulta_saldo').sum()),
        simulaciones_credito = ('tipo_evento',         lambda x: (x == 'simulacion_credito').sum()),
        solicitudes_soporte  = ('tipo_evento',         lambda x: (x == 'solicitud_soporte').sum()),
        pct_exitosos         = ('resultado_evento',    lambda x: (x == 'exitoso').sum() / len(x)),
        pct_abandonados      = ('resultado_evento',    lambda x: (x == 'abandonado').sum() / len(x)),
        sesion_promedio_seg  = ('duracion_sesion_seg', 'mean'),
    ).reset_index()

    # Tasa de conversión de pagos
    feats['tasa_conversion_pago'] = (
        feats['pagos_exitosos'] / feats['pagos_iniciados'].replace(0, np.nan)
    ).fillna(0)

    print(f"  [build_eventos_features] {len(feats):,} clientes con features digitales")
    return feats


# ============================================================
# 5. ENSAMBLE FINAL
# ============================================================
def build_dataset(data_path: str = 'data/',
                  save_path: str = None) -> pd.DataFrame:
    """
    Pipeline completo: carga, limpia, construye features y ensambla
    el dataset analítico listo para modelado.

    Args:
        data_path: ruta a la carpeta con los CSVs originales
        save_path: si se especifica, guarda el CSV resultante en esa ruta

    Returns:
        DataFrame analítico con todas las features y la variable objetivo.
    """
    print("=" * 55)
    print("  TUMIPAY — Pipeline de Feature Engineering")
    print("=" * 55)

    # 1. Cargar
    raw = load_raw_data(data_path)

    # 2. Limpiar
    print("\n[Limpieza]")
    clientes_clean = clean_clientes(raw['clientes'])
    creditos_clean = clean_creditos(raw['creditos'])
    pagos_clean    = clean_pagos(raw['pagos'])

    # 3. Construir target
    print("\n[Variable objetivo]")
    target = build_target(pagos_clean)

    # 4. Features digitales
    print("\n[Features digitales]")
    feats_eventos = build_eventos_features(raw['eventos'])

    # 5. Ensamblar
    print("\n[Ensamble]")
    df = creditos_clean.merge(
        target[['credito_id', 'mora_30', 'max_dias_mora_6m',
                'pct_pagadas_tiempo', 'n_cuotas_con_mora']],
        on='credito_id', how='inner'
    )
    df = df.merge(
        clientes_clean[[
            'cliente_id', 'edad', 'genero', 'estrato', 'nivel_educativo',
            'ocupacion', 'ingreso_mensual_estimado', 'score_externo',
            'canal_adquisicion', 'tiene_producto_ahorro', 'numero_dependientes',
            'dispositivo_principal', 'score_externo_nulo', 'edad_sospechosa'
        ]],
        on='cliente_id', how='left'
    )
    df = df.merge(feats_eventos, on='cliente_id', how='left')

    # Clientes sin eventos → llenar con 0
    cols_eventos = feats_eventos.columns.drop('cliente_id').tolist()
    df[cols_eventos] = df[cols_eventos].fillna(0)

    print(f"  Dataset final: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"  Tasa de mora: {df['mora_30'].mean():.2%}")
    print(f"  Nulos restantes: {df.isnull().sum().sum()}")

    # 6. Guardar si se especifica ruta
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n  Guardado en: {out.resolve()}")

    print("\n" + "=" * 55)
    print("  Pipeline completado exitosamente.")
    print("=" * 55)

    return df


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================
if __name__ == '__main__':
    # Permite correr: python src/feature_engineering.py
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/'
    save_path = sys.argv[2] if len(sys.argv) > 2 else 'data/dataset_analitico.csv'
    df = build_dataset(data_path=data_path, save_path=save_path)
    print(df.head())
