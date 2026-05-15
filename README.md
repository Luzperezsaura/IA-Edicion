# IA-Edicion
Repositorio de entrenamiento de red neuronal para la edición de estudios biomecánicos de la marcha.
En los diferentes cuadernos están las diferentes versiones de entrenamiento con los cambios que se fueron aplicando.

### Prueba 1:
Prueba de concepto con el modelo YOLO-v8 nano 
```
# Ejecutamos el entrenamiento por consola (CLI) de Ultralytics
!yolo pose train data=/content/data.yaml model=yolov8n-pose.pt epochs=100 imgsz=640 batch=16 project=biomecanica_proyecto name=modelo_ojalillos device=0
```

### Prueba 2:
Se cambia el tamaño de imagen a 1024x1024 y se modifica el calculo del bounding box para mejorar la precisión. Anteriormente la caja no incluia la totalidad de la pantorilla y algunos puntos quedaban por fuera.
```
´!yolo pose train data=/content/data.yaml model=yolov8m-pose.pt epochs=100 imgsz=1024 batch=8 project=biomecanica name=modelo_v3 device=0
```
El bounding box se modificó en el archivo basedatos.py

### Prueba 3:
Se cambia el modelo de nano a medium y se suben las epocas a 150 para mejorar la precisión de las detecciones. Ademas, se cambia el parámetro `fliplr=0.0` para que no espeje las imagenes de la base de datos.
```
!yolo pose train data=/content/data.yaml model=yolov8m-pose.pt epochs=150 imgsz=1024 batch=8 project=biomecanica name=modelo_izq_v1 device=0 fliplr=0.0
```
