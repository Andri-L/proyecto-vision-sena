import cv2
import numpy as np
import urllib.request
import supervision as sv
from ultralytics import YOLO
import time
import ctypes
import pandas as pd
from datetime import datetime

# =============================
# CONFIGURACIÓN DE FUENTE
# =============================
# Estas variables se pueden editar en el código.
streaming_url = 'http://192.168.173.144:8080/shot.jpg'
video_path = r'C:/Users/User/Desktop/demo.mp4'  # Ruta al video (archivo local)

# -----------------------------
# Selección de modo
# -----------------------------
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

# =============================
# CONFIGURACIÓN INICIAL DEL MODELO Y ZONAS
# =============================
model = YOLO('yolov8s.pt')

vehicle_mapping = {
    "car": "auto",
    "truck": "camion",
    "motorcycle": "moto",
    "bicycle": "bicicleta",
    "bus": "bus"
}

vehicle_class_ids = {cls_id for cls_id, cls_name in model.names.items() if cls_name in vehicle_mapping}
conf_threshold = 0.5

colors = sv.ColorPalette.from_hex(['#FF0000', '#00FF00'])

zones = None
zone_annotators = None
box_annotators = None
# Los contadores serán acumulativos (no se reinician cada frame)
zone_vehicle_counts = {}

# Lista para guardar los registros de cada frame
log_entries = []

# Variables para tracking
tracked_objects = {}  # Diccionario: id -> { "centroid": (x,y), "class": str, "counted_zones": set() }
next_object_id = 0

# Configurar DPI awareness en Windows (opcional)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        print("No se pudo establecer la conciencia de DPI:", e)

# =============================
# BUCLE PRINCIPAL DE PROCESAMIENTO
# =============================
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

        # Inicializar zonas (se definen una sola vez)
        if zones is None:
            # Definir los polígonos con las coordenadas proporcionadas sin transformación:
            # Polígono Rojo (nuevas coordenadas):
            poligono1_coords = np.array([(859, 274), (1051, 365), (1000, 283)], np.int32)
            # Polígono Verde (se mantiene el original):
            poligono2_coords = np.array([(319, 715), (1275, 269), (1277, 449), (884, 716)], np.int32)
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

        # -----------------------------
        # DETECCIÓN CON YOLOv8
        # -----------------------------
        results = model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)

        if detections and len(detections):
            mask = (detections.confidence > conf_threshold) & np.isin(detections.class_id, list(vehicle_class_ids))
            detections = detections[mask]
        else:
            detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,)))

        # Extraer información de detecciones para el tracker
        det_info = []
        if detections is not None and len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                cls_id = int(detections.class_id[i])
                cls_name = model.names[cls_id]
                det_info.append((centroid, bbox, cls_name))

        # --- ACTUALIZACIÓN DEL TRACKER ---
        updated_tracks = {}
        assigned_indices = set()
        # Actualizar tracks existentes
        for track_id, track in tracked_objects.items():
            min_dist = float('inf')
            best_index = None
            for i, (centroid, bbox, cls_name) in enumerate(det_info):
                dist = np.linalg.norm(np.array(centroid) - np.array(track["centroid"]))
                if dist < min_dist:
                    min_dist = dist
                    best_index = i
            if best_index is not None and min_dist < 50:  # Umbral de asociación
                updated_tracks[track_id] = {
                    "centroid": det_info[best_index][0],
                    "class": det_info[best_index][2],
                    "counted_zones": track["counted_zones"]
                }
                assigned_indices.add(best_index)
        # Agregar nuevos tracks para detecciones sin asignar
        for i, (centroid, bbox, cls_name) in enumerate(det_info):
            if i not in assigned_indices:
                updated_tracks[next_object_id] = {
                    "centroid": centroid,
                    "class": cls_name,
                    "counted_zones": set()
                }
                next_object_id += 1

        tracked_objects = updated_tracks

        # --- ACTUALIZAR CONTADORES POR ZONA A PARTIR DEL TRACKING ---
        for track in tracked_objects.values():
            # Revisar en cada zona si el objeto está presente y no fue contado previamente
            for zone_idx, zone in enumerate(zones):
                if cv2.pointPolygonTest(zone.polygon, (track["centroid"][0], track["centroid"][1]), False) >= 0:
                    if zone_idx not in track["counted_zones"]:
                        vehicle_type = vehicle_mapping.get(track["class"], None)
                        if vehicle_type is not None:
                            zone_vehicle_counts[zone_idx][vehicle_type] += 1
                        track["counted_zones"].add(zone_idx)

        # -----------------------------
        # DIBUJO DE ANOTACIONES Y CONTADORES EN CADA ZONA
        # -----------------------------
        for idx, (zone, zone_annotator, box_annotator) in enumerate(zip(zones, zone_annotators, box_annotators)):
            mask = zone.trigger(detections=detections)
            detections_in_zone = detections[mask]

            frame = box_annotator.annotate(scene=frame, detections=detections_in_zone)
            frame = zone_annotator.annotate(scene=frame)

            # Preparar el texto a mostrar (contadores acumulativos)
            text_zone_lines = [f"{k}: {v}" for k, v in zone_vehicle_counts[idx].items()]
            text_position = [(10, 30), (width - 250, 30)][idx]
            zone_name = ["Roja", "Verde"][idx]

            # Calcular dimensiones de la caja de texto
            text_size = cv2.getTextSize(f"Zona {zone_name}:", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            rect_x, rect_y = text_position
            rect_width = max(text_size[0], 200)
            rect_height = text_size[1] * (len(text_zone_lines) + 2) + 10

            # Dibujar recuadro de fondo semitransparente para el texto
            overlay = frame.copy()
            alpha = 0.5  # Nivel de transparencia
            cv2.rectangle(overlay, (rect_x - 5, rect_y - 20), (rect_x + rect_width + 5, rect_y + rect_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Dibujar el título y cada línea de texto (con salto de línea)
            cv2.putText(frame, f"Zona {zone_name}:", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, line in enumerate(text_zone_lines):
                cv2.putText(frame, line, (text_position[0], text_position[1] + 25 * (i + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Registrar en el log: timestamp y resumen de contadores por zona
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detections_summary = "; ".join(
            [f"Zona {['Roja','Verde'][i]}: " + ", ".join([f"{k}={v}" for k, v in zone_vehicle_counts[i].items()]) for i in zone_vehicle_counts]
        )
        log_entries.append({"timestamp": timestamp, "detections": detections_summary})

        cv2.imshow("IP Camera - Conteo de Vehiculos por Area", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.1)

except Exception as e:
    print("Error:", e)

if source_mode == "video":
    cap.release()
cv2.destroyAllWindows()

# Exportar el log a CSV
df_log = pd.DataFrame(log_entries)
df_log.to_csv("registro_frames.csv", index=False)
print("Registro exportado a 'registro_frames.csv'")
