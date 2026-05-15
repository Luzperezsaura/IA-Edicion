from ultralytics import YOLO #type: ignore
import os

# --- CONFIGURACIÓN ---
MODELO_PATH = 'best-3.pt' # chequear ruta (creo que quedó en descargas)
IMAGEN_PRUEBA = 'dataset_biomecanica/images/val'
UMBRAL_CONFIANZA = 0.6 # Solo acepta detecciones con más del 60% de seguridad. Ir variando umbrales

def probar_modelo():
    print(f"Cargando modelo desde: {MODELO_PATH}")
    # Ver si funciona con la CPU sino correrlo en colab con GPU
    model = YOLO(MODELO_PATH)
    
    print(f"Analizando imagen: {IMAGEN_PRUEBA}...")
    
    # Ejecutar la predicción
    # save = True guardará automáticamente una copia de la foto con los puntos dibujados
    resultados = model.predict(
        source=IMAGEN_PRUEBA, 
        conf=UMBRAL_CONFIANZA, 
        save=True,
        project='resultados_biomecanica', # Carpeta donde guardará la imagen visual
        name='prueba_pacientes'
    )
    
    # Extraer las coordenadas exactas para los cálculos
    for resultado in resultados:
        nombre_archivo = os.path.basename(resultado.path)
        print("\n RESULTADOS PARA: {nombre_archivo}")
        # Verificar si detectó al menos una pierna/zona
        if len(resultado) == 0:
            print("No se detectó la zona de medición en esta imagen.")
            continue
        
        # Iterar sobre cada objeto (pierna) detectado en la imagen
        for i, obj in enumerate(resultado):
            print("\n Objeto (Pierna/Zona) #{i+1}:")
            
            # Extraer las coordenadas de los keypoints. 
            # .cpu().numpy() lo convierte de un tensor de PyTorch a un array de Python/NumPy
            keypoints = obj.keypoints.xy.cpu().numpy()[0]
            confianzas = obj.keypoints.conf.cpu().numpy()[0]
            
            # Recordamos el orden estricto de tu CSV: pi1, pi2, ti1, ti2
            nombres_puntos = ['pi1', 'pi2', 'ti1', 'ti2']
            
            for nombre, (x, y), conf in zip(nombres_puntos, keypoints, confianzas):
                # Si x e y son 0, significa que el punto está ocluido o no lo encontró
                if x == 0 and y == 0:
                    print(f" {nombre}: No detectado/Oculto")
                else:
                    print(f" {nombre}: X={x:.1f}, Y={y:.1f} (Confianza: {conf*100:.1f}%)")

if __name__ == "__main__":
    if not os.path.exists(MODELO_PATH):
        print(f"ERROR: No se encontró el archivo {MODELO_PATH}")
    elif not os.path.exists(IMAGEN_PRUEBA):
        print(f"ERROR: No se encontró la imagen {IMAGEN_PRUEBA}")
    else:
        probar_modelo()