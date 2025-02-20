import cv2
import numpy as np
import time
import os
import urllib.request

# Rutas para los archivos YOLO
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
classes_path = "coco.names"

# URLs de descarga
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
# config_url = "https://pjreddie.com/media/files/yolov3.cfg"
classes_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"


def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Descargando {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"{filename} descargado exitosamente.")
        except Exception as e:
            print(f"Error al descargar {filename}: {e}")
            exit()  # Salir del programa si falla la descarga


# Descargar archivos si no existen
download_file(weights_url, weights_path)
# download_file(config_url, config_path)
download_file(classes_url, classes_path)


# Cargar YOLOv3 (ahora con rutas relativas)
net = cv2.dnn.readNet(weights_path, config_path)

# Cargar nombres de las clases del COCO dataset
with open(classes_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Ruta del video
video_path = r"C://Users/User/Desktop/video_dron.mp4"

# Diccionario para almacenar posiciones de vehículos en diferentes frames
vehicle_positions = {}
speed_estimates = {}
frame_time = time.time()

# Factor de conversión de píxeles a metros (ajustar según la escena)
pixels_to_meters = 0.05  

# Abrir el video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se puede abrir el video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    new_time = time.time()
    dt = new_time - frame_time  # Diferencia de tiempo entre frames
    frame_time = new_time
    
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward([net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()])
    
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_names[class_id] in ["car", "truck", "bus"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            vehicle_id = i  # ID único del vehículo detectado
            
            # Calcular la velocidad si hay una posición anterior
            if vehicle_id in vehicle_positions:
                x_prev, y_prev = vehicle_positions[vehicle_id]
                distance_pixels = np.sqrt((x - x_prev) * 2 + (y - y_prev) * 2)
                distance_meters = distance_pixels * pixels_to_meters
                speed = distance_meters / dt * 3.6  # Convertir m/s a km/h
                speed_estimates[vehicle_id] = speed
                cv2.putText(frame, f"Velocidad: {speed:.2f} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Actualizar la posición del vehículo
            vehicle_positions[vehicle_id] = (x, y)
            
            # Dibujar el cuadro alrededor del vehículo
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, class_names[class_ids[i]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Mostrar frame procesado
    cv2.imshow("Detección de Vehículos y Velocidad", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()