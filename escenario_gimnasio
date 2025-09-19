# -*- coding: utf-8 -*-
# =============================================================
# Gimnasios: Exploración + Visualización + Preparación para Clustering (sin K-Means)
# Requisitos: pandas, numpy, matplotlib, (opcional: seaborn para estilo)
# Uso:
# 1) Coloca este archivo en la misma carpeta que 'escenario_gimnasios_15.csv'
# 2) Ejecuta: python analisis_gimnasios_sin_kmeans.py
# 3) Se guardarán las imágenes y se imprimirán métricas en consola
# =============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1) Cargar datos
# -------------------------------------------------------------
df = pd.read_csv("escenario_gimnasios_15.csv", encoding="utf-8-sig")

print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores faltantes por columna:\n", df.isna().sum())

# -------------------------------------------------------------
# 2) Estadística descriptiva básica
# -------------------------------------------------------------
num_desc = df.select_dtypes(include="number").describe()
print("\nDescripción numérica (describe):\n", num_desc)

# -------------------------------------------------------------
# 3) Tendencia central para variables clave
# -------------------------------------------------------------
var1 = "Ingresos_Mensuales"
var2 = "Precio_Membresia"

def modabuena(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if not m.empty else np.nan

def imprimir_tendencia(var):
    media = df[var].mean(skipna=True)
    mediana = df[var].median(skipna=True)
    moda = modabuena(df[var])
    print(f"{var} -> media: {media:.2f}, mediana: {mediana:.2f}, moda: {moda}")

print("\n=== Indicadores de tendencia central ===")
imprimir_tendencia(var1)
imprimir_tendencia(var2)

def save_show_close(path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.show()
    plt.close()

plt.figure()
plt.boxplot(df[var1].dropna(), vert=True, labels=[var1])
plt.title(f"Boxplot — {var1}")
plt.ylabel(var1)
save_show_close("boxplot_ingresos.png")

# 4.2) Histograma de Ingresos_Mensuales
plt.figure()
plt.hist(df[var1].dropna(), bins=10)
plt.title(f"Histograma — {var1}")
plt.xlabel(var1); plt.ylabel("Frecuencia")
save_show_close("hist_ingresos.png")

# 4.3) Boxplot de Precio_Membresia
plt.figure()
plt.boxplot(df[var2].dropna(), vert=True, labels=[var2])
plt.title(f"Boxplot — {var2}")
plt.ylabel(var2)
save_show_close("boxplot_precio_membresia.png")

# 4.4) Matriz de correlación y heatmap (matplotlib)
num = df.select_dtypes(include="number")
corr = num.corr(numeric_only=True)
print("\nMatriz de correlación:\n" + corr.to_string(float_format=lambda x: f"{x:.6f}"))

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
ax.set_title("Heatmap de correlaciones")
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=ax)
save_show_close("heatmap-correlaciones.png", dpi=180)

# -------------------------------------------------------------

# -------------------------------------------------------------
features_all = df.select_dtypes(include="number").copy()

features_all = features_all.fillna(features_all.mean(numeric_only=True))

high_corr_threshold = 0.90
corr_matrix = features_all.corr()
to_drop = set()
cols = corr_matrix.columns.tolist()
for i, c1 in enumerate(cols):
    for c2 in cols[i+1:]:
        r = corr_matrix.loc[c1, c2]
        if pd.notna(r) and abs(r) >= high_corr_threshold:
            to_drop.add(c2)

X_prepared = features_all.drop(columns=list(to_drop)) if to_drop else features_all.copy()

print("\n=== Preparación para clustering")
print("Variables usadas:", list(X_prepared.columns))
if to_drop:
    print("Eliminadas por alta correlación (|r|>=0.90):", list(to_drop))
else:
    print("No se eliminaron variables por alta correlación con el umbral 0.90.")

X_prepared.to_csv("gimnasioscluster.csv", index=False, encoding="utf-8-sig")

print("\nArchivos guardados:")
print(" - boxplot_ingresos.png")
print(" - hist_ingresos.png")
print(" - boxplot_precio_membresia.png")
print(" - heatmap_correlaciones.png")
print(" - gimnasios_features_preparadas.csv")
print("\nListo.")
