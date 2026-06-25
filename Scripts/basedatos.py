import os
import shutil
import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split #type: ignore

# --- CONFIGURACIÓN ---
CSV_PATH = 'C:/Users/mluzp/Desktop/Etiquetas/datos_sin_duplicados.csv'           # Ruta al archivo CSV de las etiquetas
IMAGES_DIR = 'C:/Users/mluzp/Desktop/Etiquetas'   # Ruta a la carpeta con imagenes actuales
OUTPUT_DIR = 'dataset_biomecanica'   # Carpeta donde se creará el dataset YOLO
PADDING_BBOX = 0.40                  # 40% de margen para el Bounding Box, con 10% algunos puntos quedaban sobre la linea

# El orden de los keypoints
KPT_ORDER = ['pi1', 'pi2', 'ti1', 'ti2']

def setup_directories():
    """Crea la estructura de carpetas de YOLO vacía."""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def process_dataset():
    setup_directories()
    
    # 1. Cargar CSV (sin cabeceras según la captura)
    # Columnas: id de punto, x, y, id imagen, ancho, altura
    column_names = ['kpt_id', 'x', 'y', 'filename', 'width', 'height']
    df = pd.read_csv(CSV_PATH, header=None, names=column_names)
    
    # 2. Normalizar coordenadas keypoints
    df['x_norm'] = df['x'] / df['width']
    df['y_norm'] = df['y'] / df['height']
    
    # 3. Agrupar por imagen
    grouped = df.groupby('filename')
    
    # Lista para guardar los nombres de las imágenes procesadas correctamente
    valid_images = []
    
    # Carpeta temporal para los .txt antes del split
    temp_labels_dir = 'temp_labels'
    os.makedirs(temp_labels_dir, exist_ok=True)
    
    for filename, group in grouped:
        group = group.drop_duplicates(subset=['kpt_id'], keep='first')
        img_width = group['width'].iloc[0]
        img_height = group['height'].iloc[0]
        
        # Calcular Bounding Box envolvente de los puntos visibles
        #min_x = group['x_norm'].min()
        #max_x = group['x_norm'].max()
        #min_y = group['y_norm'].min()
        #max_y = group['y_norm'].max()
        # Calcular los límites exactos de los puntos
        min_x = group['x_norm'].min()
        max_x = group['x_norm'].max()
        min_y = group['y_norm'].min()
        max_y = group['y_norm'].max()
        
        ancho_puntos = max_x - min_x
        alto_puntos = max_y - min_y
        
        # Forzar tamaño mínimo de bounding box para evitar errores por puntos fuera de la caja (mínimo 15% del ancho de la foto y 30% del alto)
        bbox_w = max(ancho_puntos * 1.5, 0.15) 
        bbox_h = max(alto_puntos * 1.2, 0.30)
        
        bbox_x_center = min_x + (ancho_puntos / 2)
        bbox_y_center = min_y + (alto_puntos / 2)
        
        # Asegurar que la caja no se salga de los bordes de la imagen
        bbox_w = min(bbox_w, 1.0)
        bbox_h = min(bbox_h, 1.0)

        # Añadir padding para que los puntos no estén en el borde exacto de la caja
        #bbox_w = (max_x - min_x) * (1 + PADDING_BBOX)
        #bbox_h = (max_y - min_y) * (1 + PADDING_BBOX)
        
        #bbox_x_center = min_x + ((max_x - min_x) / 2)
        #bbox_y_center = min_y + ((max_y - min_y) / 2)
        
        # Asegurar que el BBox no se salga de la imagen (0.0 a 1.0)
        #bbox_w = min(bbox_w, 1.0)
        #bbox_h = min(bbox_h, 1.0)
        
        # Iniciar la línea del label con: Clase (0) xc yc w h
        label_parts = [f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"]
        
        # Diccionario rápido para buscar los keypoints de esta imagen
        kpts_dict = group.set_index('kpt_id')[['x_norm', 'y_norm']].to_dict('index')
        
        # Procesar en el orden estricto de los puntos
        for kpt in KPT_ORDER:
            if kpt in kpts_dict:
                x = kpts_dict[kpt]['x_norm']
                y = kpts_dict[kpt]['y_norm']
                label_parts.append(f"{x:.6f} {y:.6f} 2") # 2 = visible
            else:
                label_parts.append("0.000000 0.000000 0") # 0 = no visible/ausente
                
        # Guardar archivo .txt
        txt_filename = filename.replace('.png', '.txt').replace('.jpg', '.txt')
        with open(os.path.join(temp_labels_dir, txt_filename), 'w') as f:
            f.write(" ".join(label_parts) + "\n")
            
        valid_images.append(filename)
        
    # 4. Divido en conjunto de entrenamiento y conjunto de validación (80/20)
    print(f"Total imágenes a procesar: {len(valid_images)}")
    #train_imgs, val_imgs = train_test_split(valid_images, test_size=0.20, random_state=42)
    train_imgs, val_imgs = train_test_split(valid_images, test_size=0.20, shuffle=False)
    
    # 5. Estructura YOLO
    def move_files(file_list, split_name):
        for img_name in file_list:
            txt_name = img_name.replace('.png', '.txt').replace('.jpg', '.txt')
            
            # Copiar imagen
            src_img = os.path.join(IMAGES_DIR, img_name)
            dst_img = os.path.join(OUTPUT_DIR, 'images', split_name, img_name)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            
            # Mover label temporal
            src_txt = os.path.join(temp_labels_dir, txt_name)
            dst_txt = os.path.join(OUTPUT_DIR, 'labels', split_name, txt_name)
            if os.path.exists(src_txt):
                shutil.move(src_txt, dst_txt)
                
    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    
    # Limpieza
    shutil.rmtree(temp_labels_dir)
    print("¡Dataset YOLO creado exitosamente!")

if __name__ == "__main__":
    process_dataset()