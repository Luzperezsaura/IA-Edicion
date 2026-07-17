from fastapi import FastAPI, UploadFile, File, Form # type: ignore
from ultralytics import YOLO # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image, ImageFile # type: ignore
import io

# 1. Configuración de seguridad para imágenes truncadas/dañadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(title="IA - Edicion FootX")

modelos = {
    "retropie": {
        "izquierdo": YOLO('C:/Users/mluzp/Desktop/Proyecto IA/MODELOS/FINAL/best-RetroPieIzq.pt'),
        "derecho": YOLO('C:/Users/mluzp/Desktop/Proyecto IA/MODELOS/FINAL/best_retropie_derL.pt')
    },
    "antepie": {
        "izquierdo": YOLO('C:/Users/mluzp/Desktop/Proyecto IA/MODELOS/FINAL/best_antepie_izq.pt'),
        "derecho": YOLO('C:/Users/mluzp/Desktop/Proyecto IA/MODELOS/FINAL/last.pt')
    },
    "sagital": {
        "izquierdo": YOLO('C:/Users/mluzp/Desktop/Proyecto IA/MODELOS/FINAL/best-sagitallarge.pt'),
        "derecho": YOLO('C:/Users/mluzp/Desktop/Proyecto IA/MODELOS/FINAL/best-sagitallarge.pt')
    }
}

DICCIONARIO_NOMBRES = {
    "retropie": {
        "izquierdo": ["pi1", "pi2", "ti1", "ti2"],
        "derecho": ["pd1", "pd2", "td1", "td2"]    
    },
    "antepie": {
        "izquierdo": ["p1", "p2", "p3"],  
        "derecho": ["p1", "p2", "p3"]  
    },
    "sagital": {
        "izquierdo": ["p1", "p2", "p3"],
        "derecho": ["p1", "p2", "p3"]
    }
}

@app.post("/analizar_pierna")
async def analizar_imagen(plano: str = Form(...), lado: str = Form(...), file: UploadFile = File(...)):
    
    # 1. Validación de seguridad
    if plano not in modelos or lado not in modelos[plano]:
        return {"status": "error", "mensaje": "Plano o lado no reconocido."}
        
    # 2. Reparación y lectura de imagen "al vuelo" en memoria
    contents = await file.read()
    try:
        # Abrimos con PIL para reparar cabeceras corruptas
        img_pil = Image.open(io.BytesIO(contents))
        img_pil.load() 
        
        # Convertimos de PIL a formato OpenCV
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {"status": "error", "mensaje": f"La imagen está corrupta o no es válida: {str(e)}"}
    
    alto, ancho, _ = img.shape
    nombre_archivo = file.filename 
    
    # 3. Llamamos al modelo
    modelo_activo = modelos[plano][lado]
    resultados = modelo_activo.predict(source=img, conf=0.6, save=False, verbose=False)
    
    # 4. Armamos la respuesta base
    respuesta = {
        "status": "success", 
        "plano": plano, 
        "lado": lado, 
        "archivo": nombre_archivo,
        "ancho": ancho,
        "alto": alto,
        "marcadores": []
    }
    
    nombres_keypoints = DICCIONARIO_NOMBRES[plano][lado]
    
    for obj in resultados[0]:
        keypoints = obj.keypoints.xy.cpu().numpy()[0]
        confianzas = obj.keypoints.conf.cpu().numpy()[0]
        
        puntos = {}
        
        for nombre, (x, y), conf in zip(nombres_keypoints, keypoints, confianzas):
            if conf >= 0.6 and not (x == 0 and y == 0):
                puntos[nombre] = {"x": round(float(x), 2), "y": round(float(y), 2)}
                
        if puntos:
            respuesta["marcadores"].append(puntos)
            
    return respuesta