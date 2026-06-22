import os
import shutil
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from PIL import ImageFile # type: ignore # BLINDAJE CONTRA ARCHIVOS ROTOS DE KINNX
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =====================================================================
# --- CONFIGURACIÓN: PLANO SAGITAL IZQUIERDO (ARCO INTERNO) ---
# =====================================================================

# El orden estricto de los keypoints para este plano (3 puntos)
KPT_ORDER = ['p1', 'p2', 'p3']

# Carpeta de salida EXCLUSIVA para este modelo
OUTPUT_DIR = 'C:/Users/mluzp/Desktop/Proyecto IA/dataset_sagital_izq'

# Orígenes de datos (Acá vas a sumar las rutas de tu pendrive cuando vuelvas)
FUENTES_DATOS = [
    {
        'prefix': 'vluro', # Prefijo para la clínica de Villa Luro
        'csv_path': 'E:/Extraccion_VLuro/tmArcoInterno_Izq/etiquetas.csv',
        'images_dir': 'E:/Extraccion_VLuro/tmArcoInterno_Izq'
    },
    {
        'prefix': 'belgr', # Prefijo para la clínica de Belgrano
        'csv_path': 'E:/Extraccion_Belgrano/tmArcoInterno_Izq/etiquetas.csv',
        'images_dir': 'E:/Extraccion_Belgrano/tmArcoInterno_Izq'
    }
]

# =====================================================================

def setup_directories():
    """Crea la estructura de carpetas de YOLO vacía."""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def process_dataset():
    setup_directories()
    
    column_names = ['kpt_id', 'x', 'y', 'filename', 'width', 'height']
    dfs = []
    
    print("Leyendo y unificando fuentes de datos del Plano Sagital...")
    for fuente in FUENTES_DATOS:
        if not os.path.exists(fuente['csv_path']):
            print(f"⚠️ ADVERTENCIA: No se encontró el CSV en: {fuente['csv_path']}. Saltando esta fuente.")
            continue
            
        df_temp = pd.read_csv(fuente['csv_path'], header=None, names=column_names)
        
        # Mapeo físico y nombre único
        df_temp['src_img_path'] = df_temp['filename'].apply(lambda x: os.path.join(fuente['images_dir'], x))
        df_temp['filename_unico'] = fuente['prefix'] + "_" + df_temp['filename']
        
        dfs.append(df_temp)
        
    if not dfs:
        print("ERROR: No se logró cargar ninguna fuente de datos. Revisá las rutas del pendrive/disco.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    mapping_src_paths = df.set_index('filename_unico')['src_img_path'].to_dict()
    
    # Normalización matemática a escala 0.0 - 1.0
    df['x_norm'] = df['x'] / df['width']
    df['y_norm'] = df['y'] / df['height']
    
    grouped = df.groupby('filename_unico')
    valid_images = []
    
    temp_labels_dir = 'temp_labels_sagital'
    os.makedirs(temp_labels_dir, exist_ok=True)
    
    for filename_unico, group in grouped:
        group = group.drop_duplicates(subset=['kpt_id'], keep='first')
        
        # Calcular los límites del Bounding Box
        min_x = group['x_norm'].min()
        max_x = group['x_norm'].max()
        min_y = group['y_norm'].min()
        max_y = group['y_norm'].max()
        
        ancho_puntos = max_x - min_x
        alto_puntos = max_y - min_y
        
        # Tamaño mínimo de seguridad (15% ancho, 30% alto) y padding
        bbox_w = max(ancho_puntos * 1.5, 0.15) 
        bbox_h = max(alto_puntos * 1.2, 0.30)
        
        bbox_x_center = min_x + (ancho_puntos / 2)
        bbox_y_center = min_y + (alto_puntos / 2)
        
        bbox_w = min(bbox_w, 1.0)
        bbox_h = min(bbox_h, 1.0)
        
        # Clase 0 (Pierna Sagital)
        label_parts = [f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"]
        kpts_dict = group.set_index('kpt_id')[['x_norm', 'y_norm']].to_dict('index')
        
        # Ciclo sobre los 3 puntos del arco interno
        for kpt in KPT_ORDER:
            if kpt in kpts_dict:
                x = kpts_dict[kpt]['x_norm']
                y = kpts_dict[kpt]['y_norm']
                label_parts.append(f"{x:.6f} {y:.6f} 2") # 2 = visible
            else:
                label_parts.append("0.000000 0.000000 0") # 0 = ausente
                
        txt_filename = filename_unico.replace('.png', '.txt').replace('.jpg', '.txt')
        with open(os.path.join(temp_labels_dir, txt_filename), 'w') as f:
            f.write(" ".join(label_parts) + "\n")
            
        valid_images.append(filename_unico)
        
    print(f"Total imágenes unificadas (Sagital): {len(valid_images)}")
    train_imgs, val_imgs = train_test_split(valid_images, test_size=0.20, shuffle=False)
    
    def move_files(file_list, split_name):
        for filename_unico in file_list:
            txt_name = filename_unico.replace('.png', '.txt').replace('.jpg', '.txt')
            
            src_img = mapping_src_paths[filename_unico]
            dst_img = os.path.join(OUTPUT_DIR, 'images', split_name, filename_unico)
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            
            src_txt = os.path.join(temp_labels_dir, txt_name)
            dst_txt = os.path.join(OUTPUT_DIR, 'labels', split_name, txt_name)
            if os.path.exists(src_txt):
                shutil.move(src_txt, dst_txt)
                
    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    
    shutil.rmtree(temp_labels_dir)
    print("¡Dataset YOLO de Arco Interno (Izquierdo) creado exitosamente!")

if __name__ == "__main__":
    process_dataset()