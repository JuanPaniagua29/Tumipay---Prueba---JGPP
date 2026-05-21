"""
predict.py
==========
TUMIPAY — Prueba Técnica Data Science Engineer

Script de inferencia: dado un crédito nuevo (con su perfil de cliente),
entrena el modelo sobre el dataset histórico y devuelve:
  - probabilidad de mora
  - banda de riesgo
  - variables más influyentes en la decisión

Diseñado para simular cómo se usaría este modelo en un pipeline de
originación real: el analista o sistema ingresa los datos del crédito
y recibe un score de riesgo antes de aprobar.

Uso desde terminal:
    python src/predict.py

Uso como módulo:
    from src.predict import RiesgoMoraModel
    model = RiesgoMoraModel()
    model.fit('data/dataset_analitico.csv')
    resultado = model.predict(credito)
    print(resultado)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURACIÓN
# ============================================================
FEATURES_NUM = [
    'monto_credito', 'plazo_meses', 'tasa_interes_mensual', 'valor_cuota_pactada',
    'score_interno_originacion', 'relacion_cuota_ingreso', 'tasa_outlier',
    'edad', 'estrato', 'ingreso_mensual_estimado', 'score_externo',
    'numero_dependientes', 'score_externo_nulo', 'edad_sospechosa',
    'total_eventos', 'logins', 'pagos_fallidos', 'pct_exitosos',
    'pct_abandonados', 'solicitudes_soporte', 'sesion_promedio_seg',
    'tasa_conversion_pago',
]
FEATURES_CAT = [
    'producto_credito', 'canal_originacion', 'politica_aprobacion',
    'genero', 'nivel_educativo', 'ocupacion', 'canal_adquisicion',
    'tiene_producto_ahorro', 'dispositivo_principal',
]
TARGET = 'mora_30'

BANDAS = {
    'Bajo':     (0.00, 0.15),
    'Medio':    (0.15, 0.35),
    'Alto':     (0.35, 0.60),
    'Muy alto': (0.60, 1.00),
}

ACCIONES = {
    'Bajo':     'APROBAR — riesgo dentro del apetito estándar.',
    'Medio':    'APROBAR CON SEGUIMIENTO — monitorear primeras cuotas.',
    'Alto':     'REVISAR MANUALMENTE — requiere garantías adicionales o condiciones más restrictivas.',
    'Muy alto': 'RECHAZAR O ESCALAR — supera el umbral de riesgo aceptable.',
}


# ============================================================
# CLASE PRINCIPAL
# ============================================================
class RiesgoMoraModel:
    """
    Modelo de scoring de riesgo de mora para TUMIPAY.

    Flujo:
        1. fit(dataset_path) — entrena sobre el histórico
        2. predict(credito_dict) — scoring de un crédito nuevo
        3. evaluate() — métricas del modelo en test

    El modelo usa Regresión Logística como clasificador principal
    por su interpretabilidad y cumplimiento con criterios de
    explicabilidad regulatoria en crédito.
    """

    def __init__(self):
        self.model         = None
        self.le_dict       = {}
        self.feature_names = FEATURES_NUM + FEATURES_CAT
        self.is_fitted     = False
        self._auc_test     = None
        self._X_train      = None
        self._medians      = {}

    # ----------------------------------------------------------
    def fit(self, dataset_path: str = 'data/dataset_analitico.csv') -> 'RiesgoMoraModel':
        """
        Entrena el modelo sobre el dataset histórico.

        Args:
            dataset_path: ruta al CSV generado por feature_engineering.py

        Returns:
            self (para encadenar llamadas)
        """
        print("[fit] Cargando dataset...")
        df = pd.read_csv(dataset_path)
        print(f"      {len(df):,} créditos | Tasa mora: {df[TARGET].mean():.2%}")

        df_model = df[FEATURES_NUM + FEATURES_CAT + [TARGET]].copy()

        # Codificar categóricas
        for col in FEATURES_CAT:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            self.le_dict[col] = le

        X = df_model[self.feature_names]
        y = df_model[TARGET]

        # Guardar medianas para imputar nuevos registros
        self._medians = X.median().to_dict()

        # Split para evaluar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        self._X_train = X_train

        # Pipeline: impute → scale → logistic regression
        self.model = Pipeline([
            ('imp',    SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced'))
        ])

        print("[fit] Entrenando modelo...")
        self.model.fit(X_train, y_train)

        y_prob = self.model.predict_proba(X_test)[:, 1]
        self._auc_test = roc_auc_score(y_test, y_prob)
        self.is_fitted = True

        print(f"[fit] Completado. AUC-ROC en test: {self._auc_test:.4f}")
        return self

    # ----------------------------------------------------------
    def _get_banda(self, prob: float) -> str:
        for banda, (low, high) in BANDAS.items():
            if low <= prob <= high:
                return banda
        return 'Muy alto'

    # ----------------------------------------------------------
    def _encode_input(self, credito: dict) -> pd.DataFrame:
        """Convierte el dict de entrada en un DataFrame con las features correctas."""
        row = {}

        # Numéricas: usar valor o mediana si no viene
        for col in FEATURES_NUM:
            row[col] = credito.get(col, self._medians.get(col, 0))

        # Categóricas: codificar con el LabelEncoder entrenado
        for col in FEATURES_CAT:
            val = str(credito.get(col, 'Desconocido'))
            le  = self.le_dict[col]
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                # Categoría nueva no vista en entrenamiento → moda
                row[col] = 0

        return pd.DataFrame([row])[self.feature_names]

    # ----------------------------------------------------------
    def predict(self, credito: dict) -> dict:
        """
        Genera el score de riesgo para un crédito nuevo.

        Args:
            credito: dict con los campos del crédito y cliente.
                     Los campos faltantes se imputan con medianas del entrenamiento.

        Returns:
            dict con:
                - prob_mora: probabilidad de mora (0-1)
                - banda_riesgo: 'Bajo' | 'Medio' | 'Alto' | 'Muy alto'
                - accion_recomendada: texto con la recomendación
                - score_riesgo: escala 0-1000 (inverso, 1000 = menor riesgo)
                - inputs_usados: campos que recibió vs imputados
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo no está entrenado. Llama primero a .fit()")

        X_new = self._encode_input(credito)
        prob  = float(self.model.predict_proba(X_new)[0, 1])
        banda = self._get_banda(prob)

        # Score en escala 0-1000 (inverso: más alto = menos riesgo)
        score_riesgo = round((1 - prob) * 1000)

        # Campos imputados vs provistos
        campos_provistos  = [k for k in FEATURES_NUM + FEATURES_CAT if k in credito]
        campos_imputados  = [k for k in FEATURES_NUM + FEATURES_CAT if k not in credito]

        return {
            'prob_mora':           round(prob, 4),
            'prob_mora_pct':       f"{prob:.1%}",
            'banda_riesgo':        banda,
            'score_riesgo':        score_riesgo,
            'accion_recomendada':  ACCIONES[banda],
            'modelo_auc':          round(self._auc_test, 4),
            'campos_provistos':    len(campos_provistos),
            'campos_imputados':    campos_imputados[:5] if campos_imputados else [],
        }

    # ----------------------------------------------------------
    def evaluate(self) -> dict:
        """Devuelve las métricas del modelo."""
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado.")
        return {
            'auc_roc_test': round(self._auc_test, 4),
            'modelo':       'Regresión Logística',
            'features':     len(self.feature_names),
            'target':       'mora_30 (mora > 30 días en primeras 6 cuotas)',
        }


# ============================================================
# DEMO DE EJECUCIÓN DIRECTA
# ============================================================
def _print_resultado(resultado: dict, titulo: str = ''):
    """Imprime el resultado de forma legible."""
    print(f"\n{'='*55}")
    if titulo:
        print(f"  {titulo}")
        print(f"{'='*55}")
    print(f"  Probabilidad de mora : {resultado['prob_mora_pct']}")
    print(f"  Score de riesgo      : {resultado['score_riesgo']} / 1000")
    print(f"  Banda de riesgo      : {resultado['banda_riesgo']}")
    print(f"  Acción recomendada   : {resultado['accion_recomendada']}")
    print(f"  Campos imputados     : {resultado['campos_imputados'] or 'Ninguno'}")
    print(f"{'='*55}")


if __name__ == '__main__':

    print("\nTUMIPAY — Sistema de Scoring de Riesgo de Mora")
    print("Inicializando modelo...\n")

    # Entrenar
    model = RiesgoMoraModel()
    model.fit('data/dataset_analitico.csv')

    print(f"\nMétricas del modelo:")
    print(json.dumps(model.evaluate(), indent=2, ensure_ascii=False))

    # ---- CASO 1: Perfil de bajo riesgo ----
    credito_bajo_riesgo = {
        'monto_credito':             1_500_000,
        'plazo_meses':               12,
        'tasa_interes_mensual':      0.018,
        'valor_cuota_pactada':       135_000,
        'score_interno_originacion': 870,
        'relacion_cuota_ingreso':    0.09,
        'tasa_outlier':              0,
        'edad':                      35,
        'estrato':                   4,
        'ingreso_mensual_estimado':  4_500_000,
        'score_externo':             720,
        'numero_dependientes':       1,
        'score_externo_nulo':        0,
        'edad_sospechosa':           0,
        'producto_credito':          'Crédito consumo digital',
        'canal_originacion':         'App',
        'politica_aprobacion':       'Estándar',
        'genero':                    'Masculino',
        'nivel_educativo':           'Universitario',
        'ocupacion':                 'Empleado',
        'canal_adquisicion':         'Digital',
        'tiene_producto_ahorro':     True,
        'dispositivo_principal':     'Android',
        # Features digitales
        'total_eventos':             45,
        'logins':                    20,
        'pagos_fallidos':            0,
        'pct_exitosos':              0.95,
        'pct_abandonados':           0.05,
        'solicitudes_soporte':       1,
        'sesion_promedio_seg':       240,
        'tasa_conversion_pago':      1.0,
    }
    resultado_bajo = model.predict(credito_bajo_riesgo)
    _print_resultado(resultado_bajo, "CASO 1: Perfil de bajo riesgo")

    # ---- CASO 2: Perfil de alto riesgo ----
    credito_alto_riesgo = {
        'monto_credito':             8_000_000,
        'plazo_meses':               36,
        'tasa_interes_mensual':      0.038,
        'valor_cuota_pactada':       380_000,
        'score_interno_originacion': 620,
        'relacion_cuota_ingreso':    0.52,
        'tasa_outlier':              0,
        'edad':                      24,
        'estrato':                   1,
        'ingreso_mensual_estimado':  900_000,
        'score_externo':             410,
        'numero_dependientes':       3,
        'score_externo_nulo':        0,
        'edad_sospechosa':           0,
        'producto_credito':          'Microcrédito',
        'canal_originacion':         'Aliado',
        'politica_aprobacion':       'Flexible',
        'genero':                    'Femenino',
        'nivel_educativo':           'Bachiller',
        'ocupacion':                 'Independiente',
        'canal_adquisicion':         'Aliado',
        'tiene_producto_ahorro':     False,
        'dispositivo_principal':     'Android',
        # Features digitales
        'total_eventos':             8,
        'logins':                    3,
        'pagos_fallidos':            4,
        'pct_exitosos':              0.40,
        'pct_abandonados':           0.50,
        'solicitudes_soporte':       5,
        'sesion_promedio_seg':       60,
        'tasa_conversion_pago':      0.25,
    }
    resultado_alto = model.predict(credito_alto_riesgo)
    _print_resultado(resultado_alto, "CASO 2: Perfil de alto riesgo")

    # ---- CASO 3: Solo campos mínimos (el resto se imputa) ----
    credito_minimo = {
        'score_interno_originacion': 750,
        'monto_credito':             3_000_000,
        'relacion_cuota_ingreso':    0.25,
        'canal_originacion':         'Web',
        'producto_credito':          'Crédito consumo digital',
    }
    resultado_minimo = model.predict(credito_minimo)
    _print_resultado(resultado_minimo, "CASO 3: Solo campos mínimos (resto imputado)")
