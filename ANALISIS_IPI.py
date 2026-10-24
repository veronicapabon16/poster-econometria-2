# %% Importación de paquetes --------------------------------------------------

# Trabajar con rutas relativas en python
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from scipy.stats import jarque_bera, probplot





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
#%%




#%%-------------------------------------------------------------------------
# MODELO ARIMA LOG+DIFERENCIADO+DESETACIONALIZADO+DUMMY
#--------------------------------------------------------------------------

# %% Extracción de la serie objetivo y limpieza numérica -----------------------
# Extracción de la columna 'Producción real' para el sector T_EG
serie_ipi = pd.to_numeric(df['Producción real'], errors='coerce').dropna() #Esto extrae la serie cuantitativa de la v. de interés, limpiando los datos nulos (0)
# %% Inspección y estadísticas descriptivas -----------------------------------
print("=== INSPECCIÓN INICIAL DE LA SERIE ===")
print(f"Tipo de objeto: {type(serie_ipi)}")
print(f"Rango de fechas: {serie_ipi.index.min().strftime('%Y-%m')} a {serie_ipi.index.max().strftime('%Y-%m')}")
print(f"Número de observaciones (T): {len(serie_ipi)}")

# Estadísticas descriptivas de la serie
print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(serie_ipi.describe())


#%%-----------------------------------------------------------------
# PASO 1: IDENTIFICACION
#------------------------------------------------------------------
# %% ==========================================================================
# GRAFICA DE LA SERIE ORIGINAL DEL IPI EN NIVEL (SIN MODIFICACIONES)

# Configuración de la figura
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

# Graficar la serie en nivel original
ax.plot(serie_ipi.index, serie_ipi.values, color='#1e3a8a', linewidth=1.5, label='IPI Original')

# Personalización del gráfico
ax.set_title('Índice de Producción Industrial (Serie Original en Nivel $Y_t$)', fontsize=12, fontweight='bold', pad=12)
ax.set_ylabel('Índice (Base DANE)', fontsize=10)
ax.set_xlabel('Fecha', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()

# Exportar la imagen en alta resolución para el póster
plt.savefig('serie_original_ipi.png', bbox_inches='tight', dpi=300)
plt.show()

print("\n -> ¡ÉXITO! Se generó 'serie_original_ipi.png' correctamente.")
# %%
# %% ==========================================================================
# FAC Y FACP DE LA SERIE ORIGINAL EN NIVEL (Y_t)

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Crear la figura con 2 subgráficos acoplados
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

# 1. Función de Autocorrelación (FAC) de la Serie Original
plot_acf(serie_ipi, lags=36, ax=axes[0], color='#1e3a8a', vlines_kwargs={'colors': '#1e3a8a'})
axes[0].set_title('A. FAC - Serie Original en Nivel ($Y_t$)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Rezago (Lags)', fontsize=10)
axes[0].set_ylabel('Autocorrelación', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. Función de Autocorrelación Parcial (FACP) de la Serie Original
plot_pacf(serie_ipi, lags=36, ax=axes[1], color='#1e3a8a', vlines_kwargs={'colors': '#1e3a8a'}, method='ywm')
axes[1].set_title('B. FACP - Serie Original en Nivel ($Y_t$)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Rezago (Lags)', fontsize=10)
axes[1].set_ylabel('Autocorrelación Parcial', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Exportar la imagen en alta resolución para el póster
plt.savefig('fac_facp_serie_original.png', bbox_inches='tight', dpi=300)
plt.show()

print("\n -> ¡ÉXITO! Se generó 'fac_facp_serie_original.png' correctamente.")
#%%

#TRANSFORMACIONES A PARTIR DE LO OBSERVADO==================================================================================
#%%Transformación logarítmica para estabilizar la varianza
serie_ipi_log = np.log(serie_ipi)
# Descomposición STL sobre la serie en logaritmo
serie_log = np.log(serie_ipi)
stl_decomp = STL(serie_log, seasonal=13).fit()
serie_sa = stl_decomp.trend + stl_decomp.resid  # Serie desestacionalizada (IPI_SA)


# Gráfica de la serie de tiempo en logaritmos
plt.figure(figsize=(10, 5))
plt.plot(serie_ipi_log, color='navy', linewidth=1.2)
plt.title("Evolución Temporal IPI (Transformación Logarítmica)")
plt.xlabel("Fecha")
plt.ylabel("ln(IPI)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%

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

# %% Diferenciación de la Serie --------------------------------------

#  Primera diferencia de la serie
serie_ipi_diff = serie_ipi_log.diff().dropna()

# Gráfica de la serie diferenciada
plt.figure(figsize=(10, 5))
plt.plot(serie_ipi_diff, color='green')
plt.title("Evolución Temporal IPI - Primera Diferencia")
plt.xlabel("Fecha")
plt.ylabel("Diferencia")
plt.grid(True)
plt.show()


#%%
# -----------------------------------------------------------------------------
# IMPORTAR IMAGEN A CARPETA: SERIE TEMPORAL: NIVEL (Y_t) Y TASA DE CRECIMIENTO (Δ ln Y_t)
# -----------------------------------------------------------------------------
# Esto nos permite evaluar visualmente la tendencia estocástica y la 
# estacionalidad en la serie original, y comprobar que la primera diferencia 
# logarítmica vuelve la serie estacionaria (representando la tasa de crecimiento).

# Crear una figura con 2 subgráficos alineados verticalmente que comparten el eje X (fechas)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300)

# Gráfico de arriba: Serie en Nivel original del IPI
axes[0].plot(serie_ipi, color='#1e3a8a', linewidth=1.5)  # Azul oscuro académico
axes[0].set_title("A. Índice de Producción Industrial (Serie Original en Nivel Y_t)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Índice (Base DANE)")
axes[0].grid(True, linestyle='--', alpha=0.5)  # Rejilla tenue para facilitar la lectura

# Gráfico de abajo: Primera diferencia logarítmica (Tasa de crecimiento mensual)
axes[1].plot(serie_ipi_log.diff().dropna(), color='#0284c7', linewidth=1.2)  # Azul medio
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)  # Línea de referencia en cero
axes[1].set_title("B. Tasa de Crecimiento Mensual Aproximada (Δ ln Y_t)", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Variación Logarítmica")
axes[1].set_xlabel("Fecha")
axes[1].grid(True, linestyle='--', alpha=0.5)

# Ajustar márgenes para evitar superposición de títulos y guardar en la raíz del proyecto
plt.tight_layout()
plt.savefig(BASE_DIR / "1_serie_temporal_ipi.png", dpi=300, bbox_inches='tight')
plt.close()  # Cerrar la figura para liberar memoria RAM
#%%


## %% ==========================================================================
# FAC Y FACP - SERIE DIFERENCIADA  Y DESESTACIONALIZADA 

# Transformaciones: Logaritmo + Descomposición STL + Primera Diferencia
serie_log = np.log(serie_ipi)
stl_decomp = STL(serie_log, seasonal=13).fit()
serie_ipi_sa = stl_decomp.trend + stl_decomp.resid  # Serie desestacionalizada
serie_diff = serie_ipi_sa.diff().dropna()           # Serie diferenciada

# Configurar figura gemela (1 fila, 2 columnas)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# A. FAC
plot_acf(
    serie_diff,
    ax=axes[0],
    lags=25,
    color='#1e3a8a',
    vlines_kwargs={'colors': '#1e3a8a'},
    title="FAC Muestral (Serie Diferenciada Desestacionalizada)"
)
axes[0].set_xlabel("Rezagos (Lags Mensuales)", fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.5)

# B. FACP
plot_pacf(
    serie_diff,
    ax=axes[1],
    lags=25,
    color='#1e3a8a',
    vlines_kwargs={'colors': '#1e3a8a'},
    method='ywm',
    title="FACP Muestral (Serie Diferenciada Desestacionalizada)"
)
axes[1].set_xlabel("Rezagos (Lags Mensuales)", fontsize=9)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Guardar la imagen en la carpeta del proyecto
plt.savefig('2_fac_facp_identificacion.png', bbox_inches='tight', dpi=300)
plt.show()

print("\n -> ¡ÉXITO! Se guardó '2_fac_facp_identificacion.png' correctamente.")
# %%


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
# PASO 2: ESTIMACIÓN FORMAL DEL MODELO OPTIMO ARIMA(1,1,1) + DUMMY COVID
# =============================================================================
print("\n=== PASO 2: ESTIMACIÓN FORMAL DEL MODELO ARIMA(1,1,1) CON VARIABLE DUMMY COVID-19 ===")

# Creación de la Variable Dummy de Intervención (Control de Shock Pandemia)
# Toma valor 1 de Marzo a Mayo de 2020 y 0 en los demás periodos
dummy_covid = pd.Series(0, index=serie_ipi.index, name='dummy_covid')
dummy_covid.loc['2020-03-01':'2020-05-01'] = 1



# Especificación y Ajuste del Modelo ARIMA(1, 1, 1) con Variable Exógena
modelo_arima_optimo = SARIMAX(
    serie_sa,
    exog=dummy_covid,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),  # Sin parte estacional en SARIMAX porque la serie ya se desestacionalizó con STL
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Ajuste / Estimación del modelo por Máxima Verosimilitud
fit_arima_optimo = modelo_arima_optimo.fit(disp=False)

# Imprimir la tabla de coeficientes y criterios de información
print(fit_arima_optimo.summary())


# %% ==========================================================================
# PASO 3: TABLA DE SUPUESTOS DE LOS RESIDUOS Y EXPORTACIÓN A PNG
# =============================================================================

print("\n=== PASO 3: EVALUACIÓN DE SUPUESTOS SOBRE LOS RESIDUOS ===")

# Extraer los residuos recortando los 2 primeros valores por d=1
residuos_clean = fit_arima_optimo.resid.iloc[2:]

#  Pruebas de Diagnóstico
# Test de Ljung-Box
lb_test = acorr_ljungbox(residuos_clean, lags=[12], return_df=True)
p_val_lb = lb_test['lb_pvalue'].iloc[0]

# Test ARCH-LM
_, p_val_arch, _, _ = het_arch(residuos_clean)

# Test Jarque-Bera
_, p_val_jb = jarque_bera(residuos_clean)

#  Creación del DataFrame de Resumen
datos_diagnostico = [
    {
        "Supuesto Evaluado": "No Autocorrelación (Ljung-Box)",
        "Estadístico / Prueba": "Ljung-Box (Lag 12)",
        "p-valor": f"{p_val_lb:.4f}",
        "Resultado": "CUMPLE" if p_val_lb > 0.05 else "NO CUMPLE"
    },
    {
        "Supuesto Evaluado": "Homocedasticidad (ARCH)",
        "Estadístico / Prueba": "Test LM ARCH",
        "p-valor": f"{p_val_arch:.4f}",
        "Resultado": "CUMPLE" if p_val_arch > 0.05 else "NO CUMPLE"
    },
    {
        "Supuesto Evaluado": "Normalidad (Jarque-Bera)",
        "Estadístico / Prueba": "Jarque-Bera",
        "p-valor": f"{p_val_jb:.4f}",
        "Resultado": "CUMPLE" if p_val_jb > 0.05 else "NO CUMPLE"
    }
]

df_tabla_supuestos = pd.DataFrame(datos_diagnostico)
print(df_tabla_supuestos.to_string(index=False))

#  Renderizado y Estilizado de la Tabla para el Póster
fig, ax = plt.subplots(figsize=(8, 2.2), dpi=300)
ax.axis('off')

tabla = ax.table(
    cellText=df_tabla_supuestos.values,
    colLabels=df_tabla_supuestos.columns,
    cellLoc='center',
    loc='center'
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.8)

# Colores de encabezados y resultados
for (row, col), cell in tabla.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1f77b4')  # Azul institucional
    else:
        if col == 3:  # Columna "Resultado"
            texto = cell.get_text().get_text()
            if texto == "CUMPLE":
                cell.set_facecolor('#d4edda')      # Verde claro
                cell.set_text_props(weight='bold', color='#155724')
            else:
                cell.set_facecolor('#f8d7da')      # Rojo claro
                cell.set_text_props(weight='bold', color='#721c24')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 0 else '#ffffff')

plt.title("DIAGNÓSTICO DE SUPUESTOS DE LOS RESIDUOS (MODELO ARIMA)", fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()

# 5. Guardar la imagen en la carpeta raíz del proyecto
plt.savefig('tabla_supuestos_residuos.png', bbox_inches='tight', dpi=300)
plt.close()

print("\n -> ¡ÉXITO! La tabla limpia se guardó como imagen en 'tabla_supuestos_residuos.png'")


#%%


# %% ==========================================================================
# PASO 4: PRONÓSTICO FUERA DE MUESTRA Y GENERACIÓN DE IMÁGENES PARA EL PÓSTER
# =============================================================================

print("\n=== PASO 4: GENERANDO PRONÓSTICO Y TABLA FORMATEADA ===")

# 0. Reasegurar modelo estimado e insumos de pronóstico
serie_log = np.log(serie_ipi)
stl_decomp = STL(serie_log, seasonal=13).fit()
serie_sa = stl_decomp.trend + stl_decomp.resid

dummy_covid = pd.Series(0, index=serie_ipi.index, name='dummy_covid')
dummy_covid.loc['2020-03-01':'2020-05-01'] = 1

modelo_optimo = SARIMAX(
    serie_sa,
    exog=dummy_covid,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit_optimo = modelo_optimo.fit(disp=False)

# Pasos a pronosticar (12 meses fuera de muestra)
horizonte = 12
exog_forecast = pd.Series(0, index=pd.date_range(start=serie_ipi.index[-1] + pd.DateOffset(months=1), periods=horizonte, freq='MS'))

# Obtención del pronóstico formal
forecast_res = fit_optimo.get_forecast(steps=horizonte, exog=exog_forecast)
df_forecast = forecast_res.summary_frame(alpha=0.05)


#GRÁFICO DE PRONÓSTICO 

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

# Mostrar los últimos 36 meses históricos
historico_reciente = serie_sa.iloc[-36:]

ax.plot(historico_reciente.index, historico_reciente.values, color='#475569', label='Histórico desestacionalizado (últimos 36 meses)')
ax.plot(df_forecast.index, df_forecast['mean'], color='#eab308', linestyle='--', marker='o', markersize=4, label='Pronóstico puntual')
ax.fill_between(
    df_forecast.index,
    df_forecast['mean_ci_lower'],
    df_forecast['mean_ci_upper'],
    color='#fef08a',
    alpha=0.5,
    label='Intervalo de predicción del 95%'
)

ax.set_xlabel('Fecha', fontsize=8)
ax.set_ylabel('ln(IPI desestacionalizado)', fontsize=8)
ax.tick_params(axis='both', labelsize=7)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=6.5, frameon=True)

plt.tight_layout()
plt.savefig('grafico_pronostico_ipi.png', bbox_inches='tight', dpi=300)
plt.close()

# TABLA DE PRONÓSTICO 

df_tabla_pronostico = pd.DataFrame({
    'Fecha': df_forecast.index.strftime('%Y-%m'),
    'Pronóstico ln(IPI SA)': df_forecast['mean'].apply(lambda x: f"{x:.4f}"),
    'Límite Inferior ln (95%)': df_forecast['mean_ci_lower'].apply(lambda x: f"{x:.4f}"),
    'Límite superior ln (95%)': df_forecast['mean_ci_upper'].apply(lambda x: f"{x:.4f}")
})

fig_t, ax_t = plt.subplots(figsize=(7, 4.5), dpi=300)
ax_t.axis('off')

tabla = ax_t.table(
    cellText=df_tabla_pronostico.values,
    colLabels=df_tabla_pronostico.columns,
    cellLoc='center',
    loc='center'
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(8.5)
tabla.scale(1.1, 1.5)

# Aplicar colores azul oscuro institucional
for (row, col), cell in tabla.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1e3a8a')
        cell.set_text_props(color='white', weight='bold')
    else:
        cell.set_facecolor('#f8fafc' if row % 2 == 0 else '#ffffff')

plt.tight_layout()
plt.savefig('tabla_pronostico_ipi.png', bbox_inches='tight', dpi=300)
plt.close()

print("\n -> ¡ÉXITO! Se guardaron correctamente ambas imágenes:")
print("    - 'grafico_pronostico_ipi.png'")
print("    - 'tabla_pronostico_ipi.png'")

#%%