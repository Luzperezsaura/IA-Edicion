import cv2 #type: ignore
import numpy as np #type: ignore
import os
from ultralytics import YOLO #type: ignore
from fastapi import FastAPI, File, UploadFile, Query #type: ignore
from fastapi.responses import JSONResponse #type: ignore
import uvicorn #type: ignore
from PIL import Image, ImageFile #type: ignore
import io

# ==========================
# Iniciar FastAPI
# ==========================
app = FastAPI()

# ==========================
# Cargar los modelos YOLO
# ==========================
#BASE_DIR = 'C:/Users/mluzp/Desktop/Proyecto IA/MODELOS/FINAL'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = {
    "A": YOLO(os.path.join(BASE_DIR, "best_retropie_derL.pt")), 
    "B": YOLO(os.path.join(BASE_DIR, "best-RetroPieIzq.pt")), 
    "C": YOLO(os.path.join(BASE_DIR, "best-sagitallarge.pt")), 
    "D": YOLO(os.path.join(BASE_DIR, "best_antepieDerLarge.pt")), 
    "E": YOLO(os.path.join(BASE_DIR, "best_antepie_izq.pt")), 
}

# Diccionario con el orden exacto de los nombres de los puntos que espera cada modelo
DICCIONARIO_NOMBRES = {
    "A": ["pd1", "pd2", "td1", "td2"],
    "B": ["pi1", "pi2", "ti1", "ti2"],
    "C": ["p1", "p2", "p3"],
    "D": ["p1", "p2", "p3"],
    "E": ["p1", "p2", "p3"]
}

# Permitir cargar PNG/JPG incompletos
ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_imdecode(contents: bytes):
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

# ==========================
# Endpoint para recibir imagen y devolver puntos
# ==========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    modelo: str = Query("A", description="Modelo a usar: A, B, C, D o E")
):
    try:
        # Validar modelo
        if modelo not in models:
            return JSONResponse(content={"error": f"Modelo {modelo} no existe. Usa 'A', 'B', 'C', 'D' o 'E'."}, status_code=400)

        model = models[modelo]
        nombres_keypoints = DICCIONARIO_NOMBRES[modelo]

        # Leer la imagen en memoria
        contents = await file.read()
        img = safe_imdecode(contents)

        if img is None:
            return JSONResponse(content={"error": "No se pudo leer la imagen"}, status_code=400)

        # Convertir a formato de 3 canales si fuera necesario
        if len(img.shape) == 2: 
            gray_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ==========================
        # Predicciones con Keypoints
        # ==========================
        results = model.predict(source=gray_3ch, conf=0.1, verbose=False)

        puntos = {}
        
        # Verificamos si el resultado contiene keypoints
        if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            # Extraemos coordenadas y confianzas de la primera detección
            if results[0].keypoints.xy is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy.cpu().numpy()[0]
                confianzas = results[0].keypoints.conf.cpu().numpy()[0] if results[0].keypoints.conf is not None else [1.0] * len(keypoints)

                for nombre, (x, y), conf in zip(nombres_keypoints, keypoints, confianzas):
                    # Omitimos puntos en (0,0) o con confianza muy baja si aplica
                    if not (x == 0 and y == 0):
                        puntos[nombre] = [int(x), int(y)]

        # ==========================
        # Retornar JSON
        # ==========================
        return JSONResponse(content={
            "modelo": modelo,
            "puntos": puntos
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ==========================
# Ejecutar servidor
# ==========================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)