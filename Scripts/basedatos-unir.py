import os
import shutil
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

# --- CONFIGURACIÓN DE MULTI-ORÍGENES ---
# Configurá acá abajo todas las fuentes de datos que quieras combinar.
# El 'prefix' se antepondrá al nombre de la foto para que no haya colisiones (Ej: lote1_i0111027.png)
FUENTES_DATOS = [
    {
        'prefix': 'lote1',
        'csv_path': 'C:/Users/mluzp/Desktop/Proyecto IA/IMAGENES FOOTX IA - 1 VO/tmRetroPieIzq/etiquetas.csv',
        'images_dir': 'C:/Users/mluzp/Desktop/Proyecto IA/IMAGENES FOOTX IA - 1 VO/tmRetroPieIzq'
    },
    {
        'prefix': 'lote2',
        'csv_path': 'C:/Users/mluzp/Desktop/imagenesia/tmRetroPieIzq/etiquetas.csv',
        'images_dir': 'C:/Users/mluzp/Desktop/imagenesia/tmRetroPieIzq'
    }
    # Podés seguir agregando más diccionarios acá abajo si sumás más consultorios o cámaras...
]

OUTPUT_DIR = 'C:/Users/mluzp/Desktop/Proyecto IA/dataset_biomecanica_total'
KPT_ORDER = ['pi1', 'pi2', 'ti1', 'ti2']

def setup_directories():
    """Crea la estructura de carpetas de YOLO vacía."""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def process_dataset():
    setup_directories()
    
    column_names = ['kpt_id', 'x', 'y', 'filename', 'width', 'height']
    dfs = []
    
    # 1. Cargar, etiquetar con prefijo y unificar todos los CSVs
    print("Leyendo y unificando fuentes de datos...")
    for fuente in FUENTES_DATOS:
        if not os.path.exists(fuente['csv_path']):
            print(f"⚠️ ADVERTENCIA: No se encontró el CSV en: {fuente['csv_path']}. Saltando esta fuente.")
            continue
            
        df_temp = pd.read_csv(fuente['csv_path'], header=None, names=column_names)
        
        # GUARDAMOS LA MAGIA: Guardar de qué carpeta física viene realmente la imagen
        df_temp['src_img_path'] = df_temp['filename'].apply(lambda x: os.path.join(fuente['images_dir'], x))
        
        # CREAMOS EL NOMBRE ÚNICO: Anteponemos el prefijo del lote para evitar colisiones
        df_temp['filename_unico'] = fuente['prefix'] + "_" + df_temp['filename']
        
        dfs.append(df_temp)
        
    if not dfs:
        print("ERROR: No se logró cargar ninguna fuente de datos válida. Revisá las rutas de configuración.")
        return
        
    # Concatenamos todos los lotes en un único DataFrame maestro
    df = pd.concat(dfs, ignore_index=True)
    
    # Creamos un diccionario de mapeo rápido: { nombre_unico: ruta_origen_real }
    # Esto le permitirá a la función move_files saber de dónde copiar cada foto más adelante
    mapping_src_paths = df.set_index('filename_unico')['src_img_path'].to_dict()
    
    # 2. Normalizar coordenadas keypoints
    df['x_norm'] = df['x'] / df['width']
    df['y_norm'] = df['y'] / df['height']
    
    # 3. Agrupar por nuestro nuevo filename único
    grouped = df.groupby('filename_unico')
    
    valid_images = []
    temp_labels_dir = 'temp_labels'
    os.makedirs(temp_labels_dir, exist_ok=True)
    
    for filename_unico, group in grouped:
        group = group.drop_duplicates(subset=['kpt_id'], keep='first')
        
        # Calcular los límites exactos de los puntos
        min_x = group['x_norm'].min()
        max_x = group['x_norm'].max()
        min_y = group['y_norm'].min()
        max_y = group['y_norm'].max()
        
        ancho_puntos = max_x - min_x
        alto_puntos = max_y - min_y
        
        # Forzar tamaño mínimo de bounding box (mínimo 15% del ancho de la foto y 30% del alto)
        bbox_w = max(ancho_puntos * 1.5, 0.15) 
        bbox_h = max(alto_puntos * 1.2, 0.30)
        
        bbox_x_center = min_x + (ancho_puntos / 2)
        bbox_y_center = min_y + (alto_puntos / 2)
        
        # Asegurar que la caja no se salga de los bordes de la imagen
        bbox_w = min(bbox_w, 1.0)
        bbox_h = min(bbox_h, 1.0)
        
        # Iniciar la línea del label con: Clase (0) xc yc w h
        label_parts = [f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"]
        
        kpts_dict = group.set_index('kpt_id')[['x_norm', 'y_norm']].to_dict('index')
        
        # Procesar en el orden estricto de los puntos
        for kpt in KPT_ORDER:
            if kpt in kpts_dict:
                x = kpts_dict[kpt]['x_norm']
                y = kpts_dict[kpt]['y_norm']
                label_parts.append(f"{x:.6f} {y:.6f} 2") # 2 = visible
            else:
                label_parts.append("0.000000 0.000000 0") # 0 = ausente
                
        # Guardar archivo .txt usando el nombre único con prefijo
        txt_filename = filename_unico.replace('.png', '.txt').replace('.jpg', '.txt')
        with open(os.path.join(temp_labels_dir, txt_filename), 'w') as f:
            f.write(" ".join(label_parts) + "\n")
            
        valid_images.append(filename_unico)
        
    # 4. Divido en conjunto de entrenamiento y validación (80/20) secuencial
    print(f"Total imágenes unificadas a procesar: {len(valid_images)}")
    train_imgs, val_imgs = train_test_split(valid_images, test_size=0.20, shuffle=False)
    
    # 5. Estructura YOLO (Modificada para leer orígenes dinámicos)
    def move_files(file_list, split_name):
        for filename_unico in file_list:
            txt_name = filename_unico.replace('.png', '.txt').replace('.jpg', '.txt')
            
            # Buscamos en el mapeo de dónde hay que sacar la imagen físicamente
            src_img = mapping_src_paths[filename_unico]
            dst_img = os.path.join(OUTPUT_DIR, 'images', split_name, filename_unico)
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            
            # Mover label temporal
            src_txt = os.path.join(temp_labels_dir, txt_name)
            dst_txt = os.path.join(OUTPUT_DIR, 'labels', split_name, txt_name)
            if os.path.exists(src_txt):
                shutil.move(src_txt, dst_txt)
                
    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    
    # Limpieza de temporales
    shutil.rmtree(temp_labels_dir)
    print("¡Dataset YOLO unificado y creado exitosamente!")

if __name__ == "__main__":
    process_dataset()