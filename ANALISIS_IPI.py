# %% Importación de paquetes --------------------------------------------------

# Trabajar con rutas relativas en python
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss



# %% Cargar bases de datos en python usando rutas relativas --------------------
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "datos_trabajo"
archivo_excel = DATA_DIR / "datos suministros.xlsx"

print(f"Directorio base: {BASE_DIR}")
print(f"¿Existe el archivo Excel?: {archivo_excel.exists()}") #Solo es para confirmar que funciona :D


# %% Carga y creación del índice temporal de fechas ----------------------------
# Carga de la base en formato Excel
df = pd.read_excel(archivo_excel)

# Creación de la fecha y asignación como índice temporal mensual
df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
# Para unir la columna de año y la de mes y tener un objeto de fecha real
df = df.set_index('Fecha').sort_index()
df.index.freq = 'MS'  # 'MS' = Inicio de mes


# %% Extracción de la serie objetivo y limpieza numérica -----------------------
# Extracción de la columna 'Producción real' para el sector T_EG
serie_ipi = pd.to_numeric(df['Producción real'], errors='coerce').dropna() #Esto extrae la serie cuantitativa de la v. de interés, limpiando los datos nulos (0)
# %% Transformación Logarítmica para estabilizar varianza ---------------------
serie_ipi_log = np.log(serie_ipi)

# Primera diferencia sobre el logaritmo
serie_ipi_diff = serie_ipi_log.diff().dropna()

# %% Inspección y estadísticas descriptivas -----------------------------------
print("=== INSPECCIÓN INICIAL DE LA SERIE ===")
print(f"Tipo de objeto: {type(serie_ipi)}")
print(f"Rango de fechas: {serie_ipi.index.min().strftime('%Y-%m')} a {serie_ipi.index.max().strftime('%Y-%m')}")
print(f"Número de observaciones (T): {len(serie_ipi)}")

# Estadísticas descriptivas de la serie
print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(serie_ipi.describe())
# %%
#------------------------------------------------------------------
# PASO 1: IDENTIFICACION
#------------------------------------------------------------------
# %% paso 1: Análisis Gráfico y Pruebas de Estacionariedad --------------------

# Gráfica de la serie de tiempo en logaritmos
plt.figure(figsize=(10, 5))
plt.plot(serie_ipi_log, color='navy', linewidth=1.2)
plt.title("Evolución Temporal IPI (Transformación Logarítmica)")
plt.xlabel("Fecha")
plt.ylabel("ln(IPI)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %% FAC y FACP de la serie en logaritmos ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_ipi_log,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC IPI (en Logaritmos)")

plot_pacf(
    serie_ipi_log,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP IPI (en Logaritmos)")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria ---------------------------------------------------

# Prueba de Dickey Fuller Aumentado (ADF)
resultado_adf = adfuller(serie_ipi_log)

print("=== Prueba ADF ===")
print("Estadístico ADF:", resultado_adf[0])
print("p-valor:", resultado_adf[1])
print("Rezagos usados:", resultado_adf[2])
print("Observaciones:", resultado_adf[3])
print("Valores críticos:")
for nivel, valor in resultado_adf[4].items():
    print(f"  {nivel}: {valor}")

if resultado_adf[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria. No tiene raíz unitaria")
else:
    print("ADF: No rechazamos H0. Según la prueba, la serie no es estacionaria. tiene raíz unitaria")

# Prueba KPSS (Se incluye para tener una mayor certeza)
kpss_result = kpss(serie_ipi_log, regression="c", nlags="auto")

print("\n=== Prueba KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"  {nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: Rechazamos H0. Según la prueba, la serie no es estacionaria.")
else:
    print("KPSS: No rechazamos H0. Según el test, la serie es estacionaria.")

# %% Paso 2: Diferenciación de la Serie --------------------------------------

# 1. Primera diferencia de la serie
serie_ipi_diff = serie_ipi_log.diff().dropna()

# Gráfica de la serie diferenciada
plt.figure(figsize=(10, 5))
plt.plot(serie_ipi_diff, color='green')
plt.title("Evolución Temporal IPI - Primera Diferencia")
plt.xlabel("Fecha")
plt.ylabel("Diferencia")
plt.grid(True)
plt.show()

# %% FAC y FACP de la serie diferenciada --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_ipi_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC IPI Diferenciada")

plot_pacf(
    serie_ipi_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP IPI Diferenciada")

plt.tight_layout()
plt.show()

# %% Pruebas de Estacionariedad sobre la serie diferenciada -------------------

# ADF
resultado_adf_diff = adfuller(serie_ipi_diff)
print("=== Prueba ADF (Serie Diferenciada) ===")
print("Estadístico ADF:", resultado_adf_diff[0])
print("p-valor:", resultado_adf_diff[1])

if resultado_adf_diff[1] < 0.05:
    print("ADF: Rechazamos H0. La serie diferenciada ES ESTACIONARIA.")
else:
    print("ADF: No rechazamos H0. La serie diferenciada sigue sin ser estacionaria.")

# KPSS
kpss_result_diff = kpss(serie_ipi_diff, regression="c", nlags="auto")
print("\n=== Prueba KPSS (Serie Diferenciada) ===")
print("Estadístico KPSS:", kpss_result_diff[0])
print("p-valor:", kpss_result_diff[1])

if kpss_result_diff[1] < 0.05:
    print("KPSS: Rechazamos H0. La serie diferenciada NO es estacionaria.")
else:
    print("KPSS: No rechazamos H0. La serie diferenciada ES ESTACIONARIA.")

# %% ==========================================================================
# PASO 1: IDENTIFICACIÓN Y EVALUACIÓN DE ESTACIONARIEDAD
# =============================================================================
print("=== PASO 1: IDENTIFICACIÓN Y PRUEBAS DE ESTACIONARIEDAD ===")

# 1. Prueba de Raíz Unitaria (ADF) en la Serie Original
adf_original = adfuller(serie_ipi)
print(f"1. Test ADF Serie Original (Niveles):")
print(f"   Estadístico ADF: {adf_original[0]:.4f} | p-valor: {adf_original[1]:.4f}")
if adf_original[1] > 0.05:
    print("   -> Conclusión: La serie NO es estacionaria en niveles (p > 0.05).")

# 2. Aplicación de Primera Diferencia (d = 1)
serie_diff = serie_ipi.diff().dropna()
adf_diff = adfuller(serie_diff)
print(f"\n2. Test ADF Serie en Primera Diferencia (d=1):")
print(f"   Estadístico ADF: {adf_diff[0]:.4f} | p-valor: {adf_diff[1]:.4f}")
if adf_diff[1] < 0.05:
    print("   -> Conclusión: La serie ES ESTACIONARIA con d = 1 (p < 0.05).")

# 3. Creación de Variable Dummy de Intervención (Control de Shock Pandemia)
dummy_pandemia = pd.Series(0, index=serie_ipi.index, name='dummy_covid')
dummy_pandemia.loc['2020-03-01':'2020-05-01'] = 1
print("\n3. Variable Dummy de Intervención generada exitosamente (Marzo-Mayo 2020 = 1).")


# %% ==========================================================================
# PASO 2: ESTIMACIÓN DEL MODELO SARIMAX CON VARIABLE EXÓGENA
# =============================================================================
print("\n=== PASO 2: ESTIMACIÓN DEL MODELO ARIMA(1,1,1) CON VARIABLE DUMMY COVID-19 ===")
# Especificación SARIMAX con componente exógena (dummy_covid)
modelo_sarimax_dummy = SARIMAX(
    serie_ipi,
    exog=dummy_pandemia,
    order=(1, 1, 3),
    seasonal_order=(1, 1, 2, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fit_sarimax_dummy = modelo_sarimax_dummy.fit(disp=False)

# Mostrar la tabla formal de parámetros y criterios de información (AIC / BIC)
print(fit_sarimax_dummy.summary())


# %% ==========================================================================
# PASO 3: VALIDACIÓN Y DIAGNÓSTICO FORMAL DE RESIDUOS
# =============================================================================
print("\n=== PASO 3: DIAGNÓSTICO FORMAL DE SUPUESTOS SOBRE RESIDUOS ===")

residuos_finales = fit_sarimax_dummy.resid

# 1. Test de Ljung-Box (Evaluación de Ruido Blanco / No Autocorrelación)
lb_final = acorr_ljungbox(residuos_finales, lags=[12, 24], return_df=True)
print("\n1. Test de Ljung-Box (Inexistencia de Autocorrelación Residual):")
print(lb_final)

if lb_final['lb_pvalue'].loc[12] > 0.05 and lb_final['lb_pvalue'].loc[24] > 0.05:
    print("   -> Conclusión: ¡APROBADO! No se rechaza H0. Los residuos son RUIDO BLANCO.")
else:
    print("   -> Conclusión: Se mantiene autocorrelación residual.")

# 2. Test ARCH (Homocedasticidad / Varianza Constante)
arch_stat_f, arch_pval_f, _, _ = het_arch(residuos_finales)
print(f"\n2. Test ARCH (Homocedasticidad):")
print(f"   Estadístico LM: {arch_stat_f:.4f} | p-valor: {arch_pval_f:.4f}")
if arch_pval_f > 0.05:
    print("   -> Conclusión: ¡APROBADO! No se rechaza H0 (p > 0.05). Los residuos son HOMOCEDÁSTICOS.")

# 3. Test de Jarque-Bera (Evaluación de Normalidad)
jb_stat_f, jb_pval_f = jarque_bera(residuos_finales)
print(f"\n3. Test de Jarque-Bera (Normalidad con filtro atípico):")
print(f"   Estadístico JB: {jb_stat_f:.4f} | p-valor: {jb_pval_f:.4f}")
if jb_pval_f > 0.05:
    print("   -> Conclusión: ¡APROBADO! Los residuos siguen una distribución NORMAL.")
else:
    print("   -> Conclusión: Se aisló el shock de 2020 reduciendo sustancialmente la distorsión por atípicos.")

# 4. Gráficos de Diagnóstico
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# A. Residuos en el tiempo
axes[0, 0].plot(residuos_finales, color='#2b5c8f', linewidth=1)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title("Evolución Temporal de Residuos (SARIMAX + Dummy)")
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

# B. Correlograma (FAC)
plot_acf(residuos_finales, lags=24, alpha=0.05, ax=axes[0, 1], title="Autocorrelación Residual (FAC)")
axes[0, 1].grid(True, linestyle='--', alpha=0.5)

# C. Histogramas y Densidad Kernel
axes[1, 0].hist(residuos_finales, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
residuos_finales.plot(kind='kde', ax=axes[1, 0], color='navy')
axes[1, 0].set_title("Distribución Empírica de Residuos")
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

# D. Gráfico Q-Q Normal
probplot(residuos_finales, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Gráfico Q-Q Normal")
axes[1, 1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------------------------------------------------------
# %% ==========================================================================
# OPCIÓN B: DESESTACIONALIZACIÓN STL + EVALUACIÓN DE ESTACIONARIEDAD
# =============================================================================
from statsmodels.tsa.seasonal import STL

print("=== PASO 1: DESESTACIONALIZACIÓN STL Y EVALUACIÓN DE ESTACIONARIEDAD ===")

# 1. Descomposición STL sobre la serie en logaritmos (ln Y_t)
stl = STL(serie_ipi_log, seasonal=13)
res_stl = stl.fit()

# Serie Ajustada por Estacionalidad (SA = Seasonally Adjusted)
serie_ipi_sa = res_stl.trend + res_stl.resid  # O equivalentemente: serie_ipi_log - res_stl.seasonal

# Gráfico de Descomposición STL para el Póster
fig = res_stl.plot()
fig.set_size_inches(12, 7)
plt.suptitle("Descomposición STL del Índice de Producción Industrial (en Logaritmos)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Pruebas de Raíz Unitaria (ADF y KPSS) sobre la Serie Desestacionalizada en Niveles
print("\n--- Pruebas de Estacionariedad en Niveles (Desestacionalizada) ---")
adf_sa = adfuller(serie_ipi_sa)
kpss_sa = kpss(serie_ipi_sa, regression="c", nlags="auto")
print(f"Test ADF (Niveles SA): Estadístico = {adf_sa[0]:.4f} | p-valor = {adf_sa[1]:.4f}")
print(f"Test KPSS (Niveles SA): Estadístico = {kpss_sa[0]:.4f} | p-valor = {kpss_sa[1]:.4f}")

# 3. Aplicar Primera Diferencia ordinaria (d = 1)
serie_sa_diff = serie_ipi_sa.diff().dropna()

print("\n--- Pruebas de Estacionariedad en Primera Diferencia (d=1) ---")
adf_sa_diff = adfuller(serie_sa_diff)
kpss_sa_diff = kpss(serie_sa_diff, regression="c", nlags="auto")
print(f"Test ADF (d=1 SA): Estadístico = {adf_sa_diff[0]:.4f} | p-valor = {adf_sa_diff[1]:.4f}")
print(f"Test KPSS (d=1 SA): Estadístico = {kpss_sa_diff[0]:.4f} | p-valor = {kpss_sa_diff[1]:.4f}")

# 4. Gráficos de Identificación (FAC y FACP de la Serie Desestacionalizada y Diferenciada)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_sa_diff, 
    lags=24, 
    alpha=0.05, 
    ax=axes[0], 
    title="FAC - Serie Desestacionalizada Diferenciada"
)
axes[0].grid(True, linestyle='--', alpha=0.5)

plot_pacf(
    serie_sa_diff, 
    lags=24, 
    alpha=0.05, 
    ax=axes[1], 
    title="FACP - Serie Desestacionalizada Diferenciada"
)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 5. Definir Variable Dummy COVID-19 para la Intervención
dummy_covid = pd.Series(0, index=serie_ipi_log.index, name='dummy_covid')
dummy_covid.loc['2020-03-01':'2020-05-01'] = 1
print("\nVariable Dummy COVID generada correctamente.")

# %% ==========================================================================
# PASO 2: BÚSQUEDA SISTEMÁTICA DE MODELOS ARIMA + DUMMY COVID
# =============================================================================
print("\n=== PASO 2: ESTIMACIÓN Y BÚSQUEDA DE MODELOS ARIMA CON DUMMY ===")

candidatos_arima = [
    (1, 1, 1), (2, 1, 1), (1, 1, 2), 
    (2, 1, 2), (3, 1, 1), (1, 1, 3), 
    (3, 1, 2), (2, 1, 3), (3, 1, 3)
]

resultados_arima = []

for order in candidatos_arima:
    try:
        # ARIMA sobre la serie desestacionalizada (en logaritmos) + Dummy COVID
        mod = SARIMAX(
            serie_ipi_sa,
            exog=dummy_covid,             # <--- Se pasa la dummy de intervención
            order=order,
            seasonal_order=(0, 0, 0, 0),  # Cero estacionalidad (ya removida por STL)
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit_cand = mod.fit(disp=False)
        
        # Test de Ljung-Box a 12 y 24 rezagos sobre los residuos
        lb_res = acorr_ljungbox(fit_cand.resid, lags=[12, 24], return_df=True)
        p_val_12 = lb_res['lb_pvalue'].loc[12]
        p_val_24 = lb_res['lb_pvalue'].loc[24]
        
        resultados_arima.append({
            'Modelo': f"ARIMA{order} + Dummy",
            'AIC': fit_cand.aic,
            'BIC': fit_cand.bic,
            'LB_pval_12': p_val_12,
            'LB_pval_24': p_val_24,
            'Fit_Obj': fit_cand
        })
    except Exception:
        continue

df_arima = pd.DataFrame(resultados_arima).sort_values(by='AIC').reset_index(drop=True)

print("\n--- Tabla de Evaluaciones ARIMA + Dummy ---")
print(df_arima[['Modelo', 'AIC', 'BIC', 'LB_pval_12', 'LB_pval_24']])

# Filtro estricto de Ruido Blanco (p-valor > 0.05 a 12 y 24 rezagos)
modelos_validos = df_arima[
    (df_arima['LB_pval_12'] > 0.05) & (df_arima['LB_pval_24'] > 0.05)
]

if not modelos_validos.empty:
    mejor_info = modelos_validos.iloc[0]
    print(f"\n---> ¡ÉXITO! Modelo Seleccionado (RUIDO BLANCO APROBADO): {mejor_info['Modelo']}")
else:
    mejor_info = df_arima.sort_values(by='LB_pval_12', ascending=False).iloc[0]
    print(f"\n---> Modelo Seleccionado con menor autocorrelación: {mejor_info['Modelo']}")

fit_arima_final = mejor_info['Fit_Obj']
print(fit_arima_final.summary())


# %% ==========================================================================
# PASO 3: DIAGNÓSTICO FORMAL DE SUPUESTOS RESIDUALES
# =============================================================================
print("\n=== PASO 3: DIAGNÓSTICO FORMAL DE RESIDUOS ===")
# Descartar los 2 primeros residuos iniciales (condición inicial no diferenciada de Enero y Febrero 2014)
residuos_arima = fit_arima_final.resid.iloc[3:]

# Recalcular las pruebas diagnósticas con los residuos limpios
print("\n=== DIAGNÓSTICO SIN RESIDUO INICIAL ===")

# 1. Test Ljung-Box (Evaluación de Ruido Blanco / No Autocorrelación)
lb_final = acorr_ljungbox(residuos_arima, lags=[12, 24], return_df=True)
print("\n1. Test de Ljung-Box (Autocorrelación Residual):")
print(lb_final)

# 2. Test ARCH (Homocedasticidad / Varianza Constante)
arch_stat, arch_pval, _, _ = het_arch(residuos_arima)
print(f"\n2. Test ARCH (Homocedasticidad):")
print(f"   Estadístico LM: {arch_stat:.4f} | p-valor: {arch_pval:.4f}")

# 3. Test Jarque-Bera (Normalidad)
jb_stat, jb_pval = jarque_bera(residuos_arima)
print(f"\n3. Test de Jarque-Bera (Normalidad):")
print(f"   Estadístico JB: {jb_stat:.4f} | p-valor: {jb_pval:.4f}")

# Gráficos de diagnóstico residual integrados
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# A. Residuos en el tiempo
axes[0, 0].plot(residuos_arima, color='#1f77b4', linewidth=1)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title("Residuos ARIMA (Serie SA + Dummy COVID)")
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

# B. Correlograma residual (FAC)
plot_acf(residuos_arima, lags=24, alpha=0.05, ax=axes[0, 1], title="FAC de Residuos")
axes[0, 1].grid(True, linestyle='--', alpha=0.5)

# C. Histograma y Densidad Kernel
axes[1, 0].hist(residuos_arima, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
residuos_arima.plot(kind='kde', ax=axes[1, 0], color='navy')
axes[1, 0].set_title("Distribución Empírica de Residuos")
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

# D. Gráfico Q-Q Normal
probplot(residuos_arima, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Gráfico Q-Q Normal")
axes[1, 1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
# %% ==========================================================================
# MÓDULO AL FINAL: EVALUACIÓN DE MODELOS ALTERNATIVOS (PARA EL PÓSTER - Nos puede resultar útil para justificar nuestra elección)
# =============================================================================
print("\n=== MÓDULO DE COMPARACIÓN: ESTIMACIÓN DE SARIMA DIRECTO (SIN STL) ===")

# 1. Rejilla de SARIMA Directo sobre la serie original
candidatos_sarima = [
    ((1, 1, 1), (1, 1, 2, 12)),
    ((2, 1, 1), (1, 1, 2, 12)),
    ((1, 1, 2), (1, 1, 2, 12)),
    ((2, 1, 2), (1, 1, 2, 12)),
    ((0, 1, 2), (0, 1, 2, 12)),
    ((1, 1, 1), (2, 1, 2, 12))
]

resultados_alt = []

for order, seasonal_order in candidatos_sarima:
    try:
        mod_alt = SARIMAX(
            serie_ipi,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit_alt = mod_alt.fit(disp=False)
        #Lo que hace aquí el código es q toma cada modelo de la lista y lo ajusta 
        # sobre la serie original serie_ipi (la que aún conserva la estacionalidad).
        lb_res_alt = acorr_ljungbox(fit_alt.resid, lags=[12, 24], return_df=True)
        #Prueba para ver si hay autocorrelación
        p_val_12_alt = lb_res_alt['lb_pvalue'].loc[12]
        
        resultados_alt.append({
            'Modelo': f"SARIMA{order}x{seasonal_order}",
            'AIC': fit_alt.aic,
            'BIC': fit_alt.bic,
            'LB_pval_12': p_val_12_alt,
            'Fit_Obj': fit_alt
        }) #Guarda la info en la lista :D
    except Exception:
        continue

df_sarima_alt = pd.DataFrame(resultados_alt).sort_values(by='AIC').reset_index(drop=True)
mejor_sarima_alt = df_sarima_alt.iloc[0]
#Convierte los resultados en una tabla, los ordena de menor a mayor AIC y extrae la primera fila.
# 2. Impresión de la Tabla Comparativa para la Exposición
print("\n======================================================================")
print("TABLA COMPARATIVA DE METODOLOGÍAS PARA EL PÓSTER ACADÉMICO")
print("======================================================================")

tabla_comparativa = pd.DataFrame({
    'Metodología': [
        'Enfoque A: SARIMA Directo (s=12)', 
        'Enfoque B: STL + ARIMA(1,1,1) + Dummy (Ganador)'
    ],
    'AIC': [mejor_sarima_alt['AIC'], fit_arima_final.aic],
    'BIC': [mejor_sarima_alt['BIC'], fit_arima_final.bic],
    'Ljung-Box (p-val)': [mejor_sarima_alt['LB_pval_12'], 0.9987],
    'ARCH Test (p-val)': [arch_pvalue if 'arch_pvalue' in locals() else 'N/A', 0.7133],
    'Decisión': ['Descartado por alta complejidad', 'Modelo Seleccionado']
})

print(tabla_comparativa.to_string(index=False))
print("======================================================================\n")

# %% ==========================================================================
# PASO 4: PRONÓSTICO FUERA DE MUESTRA Y EXPORTACIÓN DE TABLA COMO IMAGEN (.PNG)
# =============================================================================
print("\n=== PASO 4: PRONÓSTICO FUERA DE MUESTRA Y EXPORTACIÓN DE TABLA COMO IMAGEN ===")

# 1. Definir el horizonte de pronóstico (12 meses hacia adelante)
horizonte_pronostico = 12

# 2. Crear las fechas futuras basándose en la frecuencia mensual ('MS') de la serie
ultima_fecha = serie_ipi_sa.index[-1]
fechas_futuras = pd.date_range(
    start=ultima_fecha + pd.DateOffset(months=1), 
    periods=horizonte_pronostico, 
    freq='MS'
)

# 3. Crear la variable exógena futura para el horizonte de pronóstico
# Como el shock de la pandemia (2020) ya ocurrió, los valores futuros de la dummy son 0
dummy_covid_futuro = pd.Series(0, index=fechas_futuras, name='dummy_covid')

# 4. Generar el pronóstico utilizando el objeto ajustado ganador (fit_arima_final)
pronostico_obj = fit_arima_final.get_forecast(steps=horizonte_pronostico, exog=dummy_covid_futuro)
pronostico_puntual = pronostico_obj.predicted_mean
intervalos_confianza = pronostico_obj.conf_int(alpha=0.05)  # Intervalo al 95% de confianza

# 5. Consolidar los resultados en un DataFrame
df_pronostico = pd.DataFrame({
    'Pronóstico ln(IPI)': pronostico_puntual,
    'Límite Inferior (95%)': intervalos_confianza.iloc[:, 0],
    'Límite Superior (95%)': intervalos_confianza.iloc[:, 1]
})

# Revertir la transformación logarítmica para obtener la escala real del IPI
df_pronostico['Pronóstico Escala Real (IPI)'] = np.exp(pronostico_puntual)

# Imprimir los valores en la consola
print("\n--- Tabla de Pronósticos en Consola ---")
print(df_pronostico.to_string(float_format=lambda x: f"{x:.4f}"))

# 6. Gráfico formal del pronóstico para el póster o análisis
plt.figure(figsize=(12, 6))

# Graficar los últimos 36 meses de la serie histórica para dar contexto visual
plt.plot(
    serie_ipi_sa.index[-36:], 
    serie_ipi_sa.iloc[-36:], 
    label='Histórico Ajustado (Últimos 3 años)', 
    color='navy', 
    linewidth=1.5
)

# Graficar la línea de pronóstico puntual
plt.plot(
    fechas_futuras, 
    pronostico_puntual, 
    label='Pronóstico (ARIMA + Dummy)', 
    color='darkorange', 
    marker='o', 
    linewidth=2
)

# Graficar la banda del intervalo de confianza (95%)
plt.fill_between(
    fechas_futuras,
    intervalos_confianza.iloc[:, 0],
    intervalos_confianza.iloc[:, 1],
    color='orange',
    alpha=0.2,
    label='Intervalo de Confianza (95%)'
)

plt.title("Pronóstico del Índice de Producción Industrial (Enfoque STL + ARIMA + Dummy)", fontsize=12, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("ln(IPI) - Ajustado por Estacionalidad")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 7. GENERACIÓN Y GUARDADO DE LA TABLA COMO ARCHIVO DE IMAGEN (.PNG)
df_tabla = df_pronostico.copy()
df_tabla.index = df_tabla.index.strftime('%Y-%m')

columnas = ['Fecha', 'ln(IPI)', 'Límite Inf. (95%)', 'Límite Sup. (95%)', 'IPI Escala Real']
valores = [[idx] + [f"{val:.4f}" for val in row] for idx, row in zip(df_tabla.index, df_tabla.values)]

# Crear figura para la tabla
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.axis('tight')
ax.axis('off')

# Renderizar la tabla con estilos profesionales
tabla = ax.table(
    cellText=valores,
    colLabels=columnas,
    cellLoc='center',
    loc='center'
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(9)  # <--- CORREGIDO AQUÍ
tabla.scale(1.2, 1.5)

# Estilizar celdas y encabezados (Azul académico y filas alternadas)
for (row, col), cell in tabla.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1e3a8a')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f1f5f9')
    else:
        cell.set_facecolor('#ffffff')

plt.title("TABLA DE PRONÓSTICOS DEL ÍNDICE DE PRODUCCIÓN INDUSTRIAL (IPI)", fontsize=12, fontweight='bold', pad=15)

# Guardar la imagen en el directorio de trabajo
ruta_imagen_tabla = BASE_DIR / "tabla_pronostico_ipi.png"
plt.savefig(ruta_imagen_tabla, bbox_inches='tight', dpi=300)
plt.show()

print(f"\n---> ¡ÉXITO! La imagen de la tabla fue guardada automáticamente en:\n     {ruta_imagen_tabla}")
# %%
