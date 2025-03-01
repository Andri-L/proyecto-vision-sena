import math
import cv2
import numpy as np
import urllib.request
import supervision as sv
from ultralytics import YOLO
import time
import ctypes
import pandas as pd
from datetime import datetime

def initialize_tracker(distance_threshold=40, max_frames_missing=20):
    """
    Inicializa el estado del tracker con los parámetros de distancia máxima y cantidad de frames permitidos sin detección.
    
    Retorna un diccionario con:
      - center_points: {id: (x_center, y_center)} de cada objeto
      - disappeared: {id: contador de frames sin detectar}
      - id_count: contador para asignar nuevos IDs
      - distance_threshold: umbral de distancia para asociar detecciones
      - max_frames_missing: máximo de frames sin detección antes de eliminar un objeto
    """
    return {
        'center_points': {},
        'disappeared': {},
        'id_count': 1,
        'distance_threshold': distance_threshold,
        'max_frames_missing': max_frames_missing,
    }

def update_tracker(tracker_state, boxes):
    """
    Actualiza el tracker con las detecciones actuales.
    
    Parámetros:
      - tracker_state: diccionario que almacena el estado del tracker.
      - boxes: lista de detecciones en formato [x, y, w, h].
    
    Retorna una lista de objetos en formato [x, y, w, h, id], conservando el ID durante 'max_frames_missing' frames sin detección.
    """
    center_points = tracker_state['center_points']
    disappeared = tracker_state['disappeared']
    distance_threshold = tracker_state['distance_threshold']
    max_frames_missing = tracker_state['max_frames_missing']
    id_count = tracker_state['id_count']

    objects = []
    matched_ids = set()

    # Si no hay detecciones, se incrementa el contador para cada objeto y se eliminan los que excedan el límite
    if len(boxes) == 0:
        for obj_id in list(center_points.keys()):
            disappeared[obj_id] = disappeared.get(obj_id, 0) + 1
            if disappeared[obj_id] > max_frames_missing:
                del center_points[obj_id]
                del disappeared[obj_id]
        tracker_state['id_count'] = id_count
        return objects

    # Procesar cada detección
    for box in boxes:
        x, y, w, h = box
        x_center = x + w // 2
        y_center = y + h // 2

        matched = False
        # Buscar coincidencias con objetos ya rastreados
        for obj_id, point in center_points.items():
            distance = math.hypot(x_center - point[0], y_center - point[1])
            if distance < distance_threshold:
                center_points[obj_id] = (x_center, y_center)
                disappeared[obj_id] = 0
                objects.append([x, y, w, h, obj_id])
                matched_ids.add(obj_id)
                matched = True
                break

        # Si no se encontró coincidencia, se asigna un nuevo ID
        if not matched:
            center_points[id_count] = (x_center, y_center)
            disappeared[id_count] = 0
            objects.append([x, y, w, h, id_count])
            id_count += 1

    tracker_state['id_count'] = id_count

    # Incrementar el contador para objetos no emparejados y eliminarlos si exceden el límite
    for obj_id in list(center_points.keys()):
        if obj_id not in matched_ids:
            disappeared[obj_id] = disappeared.get(obj_id, 0) + 1
            if disappeared[obj_id] > max_frames_missing:
                del center_points[obj_id]
                del disappeared[obj_id]

    return objects

# CONFIGURACIÓN DE FUENTE (streaming o video)
streaming_url = 'http://192.168.173.144:8080/shot.jpg'
video_path = r'C:/Users/User/Desktop/video_dron.mp4'

print("Seleccione el modo de operación:")
print("1: Recibir imágenes vía streaming")
print("2: Ejecutar video")
modo = input("Ingrese 1 o 2: ")

if modo == "2":
    source_mode = "video"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video. Se usará el modo streaming por defecto.")
        source_mode = "streaming"
elif modo == "1":
    source_mode = "streaming"
else:
    print("Entrada no válida. Se usará el modo streaming por defecto.")
    source_mode = "streaming"

# CONFIGURACIÓN INICIAL DEL MODELO Y ZONAS
model = YOLO(r'C:\Users\User\Documents\TERCER TRIMESTRE\JAIR\proyecto\yolomaestro.pt')

vehicle_mapping = {
    "car": "auto",
    "truck": "camion",
    "motorbike": "moto",
    "bicycle": "bicicleta",
    "bus": "bus"
}

# Se filtran únicamente las clases de vehículo de interés
vehicle_class_ids = {cls_id for cls_id, cls_name in model.names.items() if cls_name in vehicle_mapping}
conf_threshold = 0.5

colors = sv.ColorPalette.from_hex(['#FF0000', '#00FF00'])

zones = None
zone_annotators = None
box_annotators = None
zone_vehicle_counts = {}

# Lista para guardar los registros de cada frame
log_entries = []

# Inicializar el tracker y el registro de zonas ya contadas para cada objeto
tracker_state = initialize_tracker(distance_threshold=25, max_frames_missing=10)
object_zones_counted = {}

# Configurar DPI awareness en Windows (opcional)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        print("No se pudo establecer la conciencia de DPI:", e)

# BUCLE PRINCIPAL DE PROCESAMIENTO
try:
    while True:
        # Capturar el frame según el modo seleccionado
        if source_mode == "streaming":
            with urllib.request.urlopen(streaming_url) as response:
                img_array = np.array(bytearray(response.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue
        else:  # Modo video
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error de lectura.")
                break

        height, width = frame.shape[:2]

        # Inicializar zonas solo una vez
        if zones is None:
            # Definir los polígonos con las coordenadas proporcionadas
            poligono1_coords = np.array([(710, 409), (838, 406), (845, 417), (712, 419)], np.int32) # red
            poligono2_coords = np.array([(458, 377), (624, 378), (630, 395), (454, 396)], np.int32) # green
            polygons = [poligono1_coords, poligono2_coords]

            zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
            zone_annotators = [
                sv.PolygonZoneAnnotator(
                    zone=zone,
                    color=colors.by_idx(idx),
                    thickness=2,
                    text_thickness=2,
                    text_scale=0.5
                )
                for idx, zone in enumerate(zones)
            ]
            box_annotators = [sv.BoxAnnotator(color=colors.by_idx(idx), thickness=2) for idx in range(len(polygons))]
            # Inicializar contadores acumulativos en cada zona
            zone_vehicle_counts = {idx: {vehicle_mapping[veh]: 0 for veh in vehicle_mapping} for idx in range(len(zones))}

        # Detección con YOLOv8
        results = model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)

        if detections and len(detections):
            mask = (detections.confidence > conf_threshold) & np.isin(detections.class_id, list(vehicle_class_ids))
            detections = detections[mask]
        else:
            detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,)))

        # Extraer información de las detecciones para el tracker
        det_boxes = []
        det_info = []  # Lista de tuplas (centro, bbox, clase)
        if detections is not None and len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                w, h = x2 - x1, y2 - y1
                det_boxes.append([int(x1), int(y1), int(w), int(h)])
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                cls_id = int(detections.class_id[i])
                cls_name = model.names[cls_id]
                det_info.append((centroid, [int(x1), int(y1), int(w), int(h)], cls_name))

        # Actualizar el tracker con las cajas detectadas
        tracked_objects = update_tracker(tracker_state, det_boxes)

        # Dibujar las cajas y mostrar el ID asignado a cada objeto
        for obj in tracked_objects:
            x, y, w, h, obj_id = obj
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Actualizar contadores por zona basados en el tracking
        for idx, obj in enumerate(tracked_objects):
            x, y, w, h, obj_id = obj
            center = (x + w // 2, y + h // 2)
            # Registrar las zonas ya contadas para cada objeto
            if obj_id not in object_zones_counted:
                object_zones_counted[obj_id] = set()

            # Obtener el tipo de vehículo a partir de la detección (si está disponible)
            if idx < len(det_info):
                vehicle_class = det_info[idx][2]
                vehicle_type = vehicle_mapping.get(vehicle_class, None)
            else:
                vehicle_type = None

            # Verificar en qué zona se encuentra el centro del objeto
            for zone_idx, zone in enumerate(zones):
                if cv2.pointPolygonTest(zone.polygon, center, False) >= 0:
                    if zone_idx not in object_zones_counted[obj_id]:
                        if vehicle_type is not None:
                            zone_vehicle_counts[zone_idx][vehicle_type] += 1
                        object_zones_counted[obj_id].add(zone_idx)

        # Dibujar anotaciones y contadores en cada zona
        for idx, (zone, zone_annotator, box_annotator) in enumerate(zip(zones, zone_annotators, box_annotators)):
            mask = zone.trigger(detections=detections)
            detections_in_zone = detections[mask]

            frame = box_annotator.annotate(scene=frame, detections=detections_in_zone)
            frame = zone_annotator.annotate(scene=frame)

            # Preparar el texto con los contadores acumulativos
            text_zone_lines = [f"{k}: {v}" for k, v in zone_vehicle_counts[idx].items()]
            text_position = [(10, 30), (width - 1100, 30)][idx]
            zone_name = ["Roja", "Verde"][idx]

            text_size = cv2.getTextSize(f"Zona {zone_name}:", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            rect_x, rect_y = text_position
            rect_width = max(text_size[0], 200)
            rect_height = text_size[1] * (len(text_zone_lines) + 2) + 10

            overlay = frame.copy()
            alpha = 0.5  # Nivel de transparencia para el rectángulo de fondo
            cv2.rectangle(overlay, (rect_x - 5, rect_y - 25),
                          (rect_x + rect_width - 50, rect_y + rect_height + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            cv2.putText(frame, f"Zona {zone_name}:", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, line in enumerate(text_zone_lines):
                cv2.putText(frame, line, (text_position[0], text_position[1] + 25 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Registrar el log con la fecha/hora y resumen de contadores por zona
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detections_summary = "; ".join(
            [f"Zona {['Roja','Verde'][i]}: " + ", ".join([f"{k}={v}" for k, v in zone_vehicle_counts[i].items()])
             for i in zone_vehicle_counts]
        )
        log_entries.append({"timestamp": timestamp, "detections": detections_summary})

        cv2.imshow("IP Camera - Conteo de Vehículos por Área", frame)
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print("Error:", e)

if source_mode == "video":
    cap.release()
cv2.destroyAllWindows()

# Exportar el log a un archivo CSV
df_log = pd.DataFrame(log_entries)
df_log.to_csv("registro_frames.csv", index=False)
print("Registro exportado a 'registro_frames.csv'")
