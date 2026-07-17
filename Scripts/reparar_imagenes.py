import os
import glob
from PIL import Image, ImageFile #type: ignore

# Permite cargar imágenes que tienen bytes faltantes
ImageFile.LOAD_TRUNCATED_IMAGES = True

# CAMBIÁ ESTA RUTA POR LA TUYA LOCAL
dataset_dir = 'C:/Users/mluzp/Desktop/Proyecto IA/IMAGENES FOOTX IA - 2vo/tmRetroPieIzq' 

# Buscamos todas las imágenes
imagenes = glob.glob(f"{dataset_dir}/**/*.png", recursive=True) + glob.glob(f"{dataset_dir}/**/*.jpg", recursive=True)

print(f"Iniciando saneamiento de {len(imagenes)} imágenes en {dataset_dir}...")
reparadas = 0

for img_path in imagenes:
    try:
        img = Image.open(img_path)
        img.load() 
        
        # Guardamos en formato RGB para asegurar compatibilidad total con YOLO
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(img_path, format="PNG") # Forzamos formato para que sea consistente
        reparadas += 1
    except Exception as e:
        print(f"No se pudo reparar {img_path}: {e}")

print(f"¡Listo! Se repararon {reparadas} imágenes.")