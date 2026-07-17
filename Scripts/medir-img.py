import pandas as pd #type: ignore
import numpy as np #type: ignore
import os

# --- CONFIGURACIÓN ---
ruta_txts = 'runs/pose/predict/labels/' # Donde YOLO guardó los .txt
ruta_csv = 'C:/Users/mluzp/Desktop/Proyecto IA/DATASETS/etiquetas_reales.csv' # Tu CSV
columnas = ['punto', 'x_real', 'y_real', 'nombre_imagen', 'ancho', 'alto']
tolerancias = [5, 10, 15] # Umbrales en píxeles para probar

# 1. Cargar datos reales
df_real = pd.read_csv(ruta_csv, header=None, names=columnas)

errores = []
resultados_detallados = []

# 2. Procesar predicciones
for archivo in os.listdir(ruta_txts):
    if not archivo.endswith('.txt'): continue
    
    with open(os.path.join(ruta_txts, archivo), 'r') as f:
        linea = f.readline().split()
        if not linea: continue
        x_norm, y_norm = float(linea[5]), float(linea[6])
    
    nombre_img = archivo.replace('.txt', '.png')
    fila = df_real[df_real['nombre_imagen'] == nombre_img]
    
    if not fila.empty:
        ancho, alto = fila['ancho'].values[0], fila['alto'].values[0]
        dist = np.sqrt(((x_norm * ancho) - fila['x_real'].values[0])**2 + 
                       ((y_norm * alto) - fila['y_real'].values[0])**2)
        errores.append(dist)
        resultados_detallados.append((nombre_img, dist))

# 3. Cálculo de métricas
mae = np.mean(errores)
print(f"--- RESULTADOS DE VALIDACIÓN ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} píxeles\n")

print(f"{'Tolerancia (px)':<18} | {'Precisión Clínica (%)'}")
print("-" * 40)
for tol in tolerancias:
    precision = (sum(1 for e in errores if e <= tol) / len(errores)) * 100
    print(f"{tol:<18} | {precision:.2f}%")

# 4. (Opcional) Guardar reporte para el informe
with open('reporte_validacion_clinica.txt', 'w') as f:
    f.write(f"Reporte de Validación Externa\nMAE: {mae:.2f} px\n\nDetalle por imagen:\n")
    for img, dist in resultados_detallados:
        f.write(f"{img}: {dist:.2f} px\n")