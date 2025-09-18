# -*- coding: utf-8 -*-
# =============================================================
# Análisis muy simple y comentado para 15 tiendas
# Requisitos: pandas, matplotlib, seaborn
# Uso:
# 1) Coloca este archivo en la misma carpeta que 'escenario_tiendas_15.csv'
# 2) Ejecuta: python analisis_tiendas_simple_comentado.py
# 3) Se guardarán las imágenes y verás las métricas impresas en consola
# =============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 1) Cargar datos
# -------------------------------------------------------------
df = pd.read_csv("escenario_tiendas_15.csv", encoding="utf-8-sig")

print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores faltantes por columna:\n", df.isna().sum())

# -------------------------------------------------------------
# 2) Estadística descriptiva básica
# -------------------------------------------------------------
desc = df.select_dtypes(include="number").describe()
print("\nDescripción numérica:\n", desc)

# -------------------------------------------------------------
# 3) Tendencia central para columnas numéricas
# -------------------------------------------------------------
def sacarmedias(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include="number").columns

    resultados = []
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            media = mediana = moda = None
            print(f"{col} -> sin datos")
        else:
            media = s.mean()
            mediana = s.median()
            moda_series = s.mode()
            moda = moda_series.iloc[0] if not moda_series.empty else None
            print(f"{col} -> media: {media:.2f}, mediana: {mediana:.2f}, moda: {moda}")

        resultados.append({"variable": col, "media": media, "mediana": mediana, "moda": moda})

    return pd.DataFrame(resultados)

print("\n=== Indicadores de tendencia central (todas las numéricas) ===")
resumen_tendencia = sacarmedias(df)

var1 = "Ventas_Mensuales"
var2 = "Precio_Promedio"

def save_show_close(path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.show()
    plt.close()

# -------------------------------------------------------------
# 4) Gráficos simples
# -------------------------------------------------------------
plt.figure()
df[var1].plot(kind="box", title=f"Boxplot - {var1}", color='red')
save_show_close("boxplot_ventas.png")

plt.figure()
df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}", color='red')
plt.xlabel("Valor"); plt.ylabel("Frecuencia")
save_show_close("hist_ventas.png")

plt.figure()
df[var2].plot(kind="box", title=f"Boxplot - {var2}", color='red')
save_show_close("boxplot_precio_promedio.png")

# -------------------------------------------------------------
# 5) Matriz de correlación y heatmap
# -------------------------------------------------------------
num = df.select_dtypes(include="number")
corr = num.corr(numeric_only=True)

print("\nMatriz de correlación:\n" + corr.to_string(float_format=lambda x: f"{x:.6f}"))

fig = plt.figure(figsize=(12, 9))         
ax = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    square=False,              
    cbar_kws={"shrink": 0.9}
)
ax.set_title("Heatmap de correlaciones (variables numéricas)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
save_show_close("heatmap_correlaciones.png", dpi=180)

# -------------------------------------------------------------
# 6) Correlaciones más fuertes con la variable objetivo
# -------------------------------------------------------------
if var1 in corr.columns:
    top_corr = corr[var1].dropna().abs().sort_values(ascending=False).head(6)
    print(f"\nCorrelaciones más fuertes (absolutas) con {var1}:\n", top_corr)

print("\nListo.")
print("Imágenes guardadas:")
print(" - boxplot_ventas.png")
print(" - hist_ventas.png")
print(" - boxplot_precio_promedio.png")
print(" - heatmap_correlaciones.png")
