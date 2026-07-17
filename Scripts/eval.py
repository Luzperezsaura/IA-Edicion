import pandas as pd #type: ignore
import numpy as np #type: ignore
import os

# 1. Cargamos tu CSV de etiquetas reales
columnas = ['punto', 'x_real', 'y_real', 'nombre_imagen', 'ancho', 'alto']
df_real = pd.read_csv('C:/Users/mluzp/Desktop/Proyecto IA/IMAGENES FOOTX IA - 2vo/tmAntePieDer/etiquetas.csv', header= None, names=columnas) # Ajusta el nombre
tolerancias = [5, 10, 15] # Umbrales en px 

# 2. Iteramos sobre los archivos TXT que generó YOLO
ruta_txts = 'C:/Users/mluzp/Desktop/Proyecto IA/runs/pose/predict-18/labels'

errores = []
resultados_detallados = []
total_imagenes = 0

for archivo in os.listdir(ruta_txts):
    with open(os.path.join(ruta_txts, archivo), 'r') as f:
        linea = f.readline().split()
        if not linea: continue
        
        # Leemos explícitamente los índices 5 y 6
        x_norm = float(linea[5])
        y_norm = float(linea[6])
    
    nombre_img = archivo.replace('.txt', '.png')
    fila = df_real[df_real['nombre_imagen'] == nombre_img]
    
    if not fila.empty:
        ancho = fila['ancho'].values[0]
        alto = fila['alto'].values[0]
        
        x_pred_px = x_norm * ancho
        y_pred_px = y_norm * alto
        
        x_real = fila['x_real'].values[0]
        y_real = fila['y_real'].values[0]
        
        dist = np.sqrt((x_pred_px - x_real)**2 + (y_pred_px - y_real)**2)
        errores.append(dist)
        
        print(f"{'Tolerancia (px)':<18} | {'Precisión Clínica (%)'}")
        print("-" * 40)
        for tol in tolerancias:
            precision = (sum(1 for e in errores if e <= tol) / len(errores)) * 100
            print(f"{tol:<18} | {precision:.2f}%")
        # DEBUG: Imprimir comparación
        print(f"Img: {nombre_img} | Pred: ({x_pred_px:.1f}, {y_pred_px:.1f}) | Real: ({x_real}, {y_real}) | Dist: {dist:.1f}")

print(f"Error Medio (MAE) del modelo: {np.mean(errores):.2f} píxeles")
# 3. Cálculo de métricas (ESTO VA FUERA DEL BUCLE)
mae = np.mean(errores)
rmse = np.sqrt(np.mean(np.square(errores)))

print(f"\n--- RESULTADOS GLOBALES PARA EL INFORME ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} px")
print(f"Error Cuadrático Medio (RMSE): {rmse: .2f} px")

# Calculamos la precisión global para 10px
tol_elegida = 8
correctas = sum(1 for e in errores if e <= tol_elegida)
precision_global = (correctas / len(errores)) * 100

print(f"Precisión Clínica (con {tol_elegida}px de tolerancia): {precision_global:.2f}%")