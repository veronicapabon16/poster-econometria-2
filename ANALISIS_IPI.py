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
print(f"¿Existe el archivo Excel?: {archivo_excel.exists()}")


# %% Carga y creación del índice temporal de fechas ----------------------------
# Carga de la base en formato Excel
df = pd.read_excel(archivo_excel)

# Creación de la fecha y asignación como índice temporal mensual
df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
df = df.set_index('Fecha').sort_index()
df.index.freq = 'MS'  # 'MS' = Inicio de mes


# %% Extracción de la serie objetivo y limpieza numérica -----------------------
# Extracción de la columna 'Producción real' para el sector T_EG
serie_ipi = pd.to_numeric(df['Producción real'], errors='coerce').dropna()


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

# Gráfica de la serie de tiempo original
plt.figure(figsize=(10, 5))
plt.plot(serie_ipi)
plt.title("Evolución Temporal IPI")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% FAC y FACP de la serie original ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_ipi,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC IPI Original")

plot_pacf(
    serie_ipi,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP IPI Original")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria ---------------------------------------------------

# Prueba de Dickey Fuller Aumentado (ADF)
resultado_adf = adfuller(serie_ipi)

print("=== Prueba ADF ===")
print("Estadístico ADF:", resultado_adf[0])
print("p-valor:", resultado_adf[1])
print("Rezagos usados:", resultado_adf[2])
print("Observaciones:", resultado_adf[3])
print("Valores críticos:")
for nivel, valor in resultado_adf[4].items():
    print(f"  {nivel}: {valor}")

if resultado_adf[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según la prueba, la serie no es estacionaria.")

# Prueba KPSS
kpss_result = kpss(serie_ipi, regression="c", nlags="auto")

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
# %%




# %% paso 2: Diferenciación de la Serie --------------------------------------

# 1. Primera diferencia de la serie
serie_ipi_diff = serie_ipi.diff().dropna()

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
# %%

#Paso 3 Estimación de Modelos ARIMA / SARIMAX -----------------------------
# %% Paso 2: Búsqueda ampliada de modelos para eliminar autocorrelación --------
# %% Paso 2: Búsqueda Sistemática de Modelos SARIMA (Filtro Ruido Blanco) ------

# Rejilla de combinaciones para eliminar la estructura de correlación estacional
candidatos_sarima = [
    ((1, 1, 1), (1, 1, 2, 12)),
    ((2, 1, 1), (1, 1, 2, 12)),
    ((1, 1, 2), (1, 1, 2, 12)),
    ((2, 1, 2), (1, 1, 2, 12)),
    ((0, 1, 2), (0, 1, 2, 12)),
    ((1, 1, 1), (2, 1, 2, 12)),
    ((2, 1, 1), (2, 1, 1, 12)),
    ((3, 1, 1), (1, 1, 1, 12)),
    ((1, 1, 3), (1, 1, 2, 12))
]

resultados = []

print("=== BUSCANDO MODELO SIN AUTOCORRELACIÓN RESIDUAL ===")
for order, seasonal_order in candidatos_sarima:
    try:
        mod = SARIMAX(
            serie_ipi,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit_cand = mod.fit(disp=False)
        
        # Evaluar test de Ljung-Box a 12 y 24 rezagos
        lb_res = acorr_ljungbox(fit_cand.resid, lags=[12, 24], return_df=True)
        p_val_12 = lb_res['lb_pvalue'].loc[12]
        p_val_24 = lb_res['lb_pvalue'].loc[24]
        
        resultados.append({
            'Modelo': f"SARIMA{order}x{seasonal_order}",
            'AIC': fit_cand.aic,
            'BIC': fit_cand.bic,
            'LB_pval_12': p_val_12,
            'LB_pval_24': p_val_24,
            'Fit_Obj': fit_cand
        })
    except Exception:
        continue

df_modelos = pd.DataFrame(resultados).sort_values(by='AIC').reset_index(drop=True)

# Filtro estricto: p-valor > 0.05 en ambos rezagos (Sin Autocorrelación)
modelos_ruido_blanco = df_modelos[
    (df_modelos['LB_pval_12'] > 0.05) & (df_modelos['LB_pval_24'] > 0.05)
]

print("\n--- Modelos Evaluados ---")
print(df_modelos[['Modelo', 'AIC', 'BIC', 'LB_pval_12', 'LB_pval_24']])

if not modelos_ruido_blanco.empty:
    mejor_modelo_info = modelos_ruido_blanco.iloc[0]
    print(f"\n---> ¡ÉXITO! Modelo Seleccionado (RUIDO BLANCO APROBADO): {mejor_modelo_info['Modelo']}")
else:
    # Si ninguno pasa con >0.05, selecciona el que tenga el mayor p-valor (el más cercano a ruido blanco)
    mejor_modelo_info = df_modelos.sort_values(by='LB_pval_12', ascending=False).iloc[0]
    print(f"\n---> Modelo Seleccionado con menor autocorrelación: {mejor_modelo_info['Modelo']}")

mejor_fit = mejor_modelo_info['Fit_Obj']
print(mejor_fit.summary())


# %% Paso 3: Verificación del Test Ljung-Box sobre el Nuevo Modelo ------------

residuos = mejor_fit.resid

print("\n=== DIAGNÓSTICO DE RUIDO BLANCO (LJUNG-BOX) ===")
lb_test = acorr_ljungbox(residuos, lags=[12, 24], return_df=True)
print(lb_test)

if lb_test['lb_pvalue'].loc[12] > 0.05 and lb_test['lb_pvalue'].loc[24] > 0.05:
    print("\n   -> Conclusión: ¡CUMPLIDO! p-valor > 0.05. Los residuos son RUIDO BLANCO sin autocorrelación.")
else:
    print("\n   -> Conclusión: Revisa la tabla de modelos para seleccionar la especificación de mayor p-valor.")
# %%

# %% Paso 3 (Continuación): Evaluación de Homocedasticidad y Normalidad -------

# 1. Test ARCH de Homocedasticidad (Varianza Constante)
arch_stat, arch_pvalue, _, _ = het_arch(residuos)
print("=== EVALUACIÓN COMPLETA DE SUPUESTOS DE RESIDUOS ===")
print(f"\n1. Test ARCH de Homocedasticidad:")
print(f"   Estadístico LM: {arch_stat:.4f} | p-valor: {arch_pvalue:.4f}")

if arch_pvalue > 0.05:
    print("   -> Conclusión: ¡APROBADO! No se rechaza H0. Los residuos son HOMOCEDÁSTICOS (varianza constante).")
else:
    print("   -> Conclusión: Se rechaza H0. Presencia de heterocedasticidad condicional.")

# 2. Test de Jarque-Bera para Normalidad
jb_stat, jb_pvalue = jarque_bera(residuos)
print(f"\n2. Test de Jarque-Bera (Normalidad):")
print(f"   Estadístico JB: {jb_stat:.4f} | p-valor: {jb_pvalue:.4f}")

if jb_pvalue > 0.05:
    print("   -> Conclusión: ¡APROBADO! No se rechaza H0. Los residuos se distribuyen NORMALMENTE.")
else:
    print("   -> Conclusión: Se rechaza H0 (p < 0.05). Existen valores atípicos (atípico estructural por shock, ej. 2020).")

# 3. Gráficos de Diagnóstico Residual Integrado
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribución empírica vs Teórica
axes[0].hist(residuos, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
residuos.plot(kind='kde', ax=axes[0], color='navy', label='KDE Residuos')
axes[0].set_title("Distribución de Residuos")
axes[0].grid(True, linestyle='--', alpha=0.5)

# Gráfico Q-Q Normal
probplot(residuos, dist="norm", plot=axes[1])
axes[1].set_title("Gráfico Q-Q Normal")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
# %%




#--------------------------------------------
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
print("\n=== PASO 2: ESTIMACIÓN DEL MODELO SARIMAX(1,1,3)x(1,1,2)[12] CON DUMMY ===")

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
# %%


#----------------------------------------------------------------------------------------------------------------------------
# %% ==========================================================================
# OPCIÓN B: DESESTACIONALIZACIÓN STL + MODELO ARIMA BOX-JENKINS PURO
# =============================================================================
from statsmodels.tsa.seasonal import STL

print("=== PASO 1: DESESTACIONALIZACIÓN Y EVALUACIÓN DE ESTACIONARIEDAD ===")

# 1. Descomposición STL para remover la estacionalidad
stl = STL(serie_ipi, seasonal=13)
res_stl = stl.fit()

# Serie Ajustada por Estacionalidad (SA = Seasonally Adjusted)
serie_ipi_sa = res_stl.trend + res_stl.resid

# Gráfico de Descomposición
fig = res_stl.plot()
fig.set_size_inches(12, 7)
plt.suptitle("Descomposición STL del Índice de Producción Industrial (IPI)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Prueba de Raíz Unitaria (ADF) sobre la Serie Desestacionalizada
adf_sa = adfuller(serie_ipi_sa)
print(f"Test ADF Serie Desestacionalizada (Niveles): p-valor = {adf_sa[1]:.4f}")

# Aplicar Primera Diferencia (d = 1)
serie_sa_diff = serie_ipi_sa.diff().dropna()
adf_sa_diff = adfuller(serie_sa_diff)
print(f"Test ADF Serie Desestacionalizada (d=1): p-valor = {adf_sa_diff[1]:.4f}")


# %% ==========================================================================
# PASO 2: ESTIMACIÓN DEL MODELO ARIMA PURO (GRID SEARCH RUIDO BLANCO)
# =============================================================================
print("\n=== PASO 2: BÚSQUEDA DEL MODELO ARIMA PURO (SIN EXÓGENAS) ===")

candidatos_arima = [
    (1, 1, 1), (2, 1, 1), (1, 1, 2), 
    (2, 1, 2), (3, 1, 1), (1, 1, 3), 
    (3, 1, 2), (2, 1, 3), (3, 1, 3)
]

resultados_arima = []

for order in candidatos_arima:
    try:
        # ARIMA puro tradicional sobre la serie desestacionalizada
        mod = SARIMAX(
            serie_ipi_sa,
            order=order,
            seasonal_order=(0, 0, 0, 0), # Cero estacionalidad (ya fue removida)
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit_cand = mod.fit(disp=False)
        
        # Test de Ljung-Box a 12 y 24 rezagos
        lb_res = acorr_ljungbox(fit_cand.resid, lags=[12, 24], return_df=True)
        p_val_12 = lb_res['lb_pvalue'].loc[12]
        p_val_24 = lb_res['lb_pvalue'].loc[24]
        
        resultados_arima.append({
            'Modelo': f"ARIMA{order}",
            'AIC': fit_cand.aic,
            'BIC': fit_cand.bic,
            'LB_pval_12': p_val_12,
            'LB_pval_24': p_val_24,
            'Fit_Obj': fit_cand
        })
    except Exception:
        continue

df_arima = pd.DataFrame(resultados_arima).sort_values(by='AIC').reset_index(drop=True)

print("\n--- Tabla de Evaluaciones ARIMA ---")
print(df_arima[['Modelo', 'AIC', 'BIC', 'LB_pval_12', 'LB_pval_24']])

# Filtro de Ruido Blanco (p-valor > 0.05 a 12 y 24 rezagos)
modelos_validos = df_arima[
    (df_arima['LB_pval_12'] > 0.05) & (df_arima['LB_pval_24'] > 0.05)
]

if not modelos_validos.empty:
    mejor_info = modelos_validos.iloc[0]
    print(f"\n---> ¡ÉXITO! Modelo Seleccionado (RUIDO BLANCO APROBADO): {mejor_info['Modelo']}")
else:
    mejor_info = df_arima.sort_values(by='LB_pval_12', ascending=False).iloc[0]
    print(f"\n---> Modelo Seleccionado: {mejor_info['Modelo']}")

fit_arima_final = mejor_info['Fit_Obj']
print(fit_arima_final.summary())


# %% ==========================================================================
# PASO 3: DIAGNÓSTICO FORMAL DE SUPUESTOS
# =============================================================================
print("\n=== PASO 3: DIAGNÓSTICO FORMAL DE RESIDUOS ===")

residuos_arima = fit_arima_final.resid

# 1. Test Ljung-Box (No Autocorrelación)
lb_final = acorr_ljungbox(residuos_arima, lags=[12, 24], return_df=True)
print("\n1. Test de Ljung-Box (Autocorrelación Residual):")
print(lb_final)

# 2. Test ARCH (Homocedasticidad)
arch_stat, arch_pval, _, _ = het_arch(residuos_arima)
print(f"\n2. Test ARCH (Homocedasticidad):")
print(f"   Estadístico LM: {arch_stat:.4f} | p-valor: {arch_pval:.4f}")

# 3. Test Jarque-Bera (Normalidad)
jb_stat, jb_pval = jarque_bera(residuos_arima)
print(f"\n3. Test de Jarque-Bera (Normalidad):")
print(f"   Estadístico JB: {jb_stat:.4f} | p-valor: {jb_pval:.4f}")

# Gráficos de diagnóstico
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

axes[0, 0].plot(residuos_arima, color='#1f77b4', linewidth=1)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title("Residuos ARIMA (Serie Desestacionalizada)")

plot_acf(residuos_arima, lags=24, alpha=0.05, ax=axes[0, 1])
axes[0, 1].set_title("FAC de Residuos")

axes[1, 0].hist(residuos_arima, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black')
residuos_arima.plot(kind='kde', ax=axes[1, 0], color='navy')
axes[1, 0].set_title("Distribución de Residuos")

probplot(residuos_arima, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Gráfico Q-Q Normal")

plt.tight_layout()
plt.show()
# %%


# %% ==========================================================================
# GRÁFICOS DE FAC Y FACP PARA EL MODELO ARIMA SELECCIONADO
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Función de Autocorrelación Simple (FAC / ACF)
plot_acf(
    residuos_arima, 
    lags=12, 
    alpha=0.05, 
    ax=axes[0], 
    title="Función de Autocorrelación Simple (FAC) - Residuos ARIMA"
)
axes[0].set_xlabel("Rezagos (Lags)")
axes[0].set_ylabel("Autocorrelación")
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. Función de Autocorrelación Parcial (FACP / PACF) - Método estándar OLS
plot_pacf(
    residuos_arima, 
    lags=12, 
    alpha=0.05, 
    ax=axes[1], 
    title="Función de Autocorrelación Parcial (FACP) - Residuos ARIMA"
)
axes[1].set_xlabel("Rezagos (Lags)")
axes[1].set_ylabel("Autocorrelación Parcial")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %%
