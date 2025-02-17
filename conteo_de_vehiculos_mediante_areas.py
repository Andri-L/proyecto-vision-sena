import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np

# --- Implementación básica de SORT ---
class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Inicializa el tracker SORT.
        Args:
            max_age (int): Máximo número de frames sin detección para considerar que un track ha terminado.
            min_hits (int): Mínimo número de detecciones consecutivas para confirmar un nuevo track.
            iou_threshold (float): Umbral de IoU para la asociación de datos.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self._next_id = 1

    def update(self, detections):
        """
        Actualiza el tracker con nuevas detecciones.
        Args:
            detections (list): Lista de detecciones en formato [[x1, y1, x2, y2, score], ...].
        Returns:
            list: Lista de tracks activos en formato [[x1, y1, x2, y2, id], ...].
        """
        if not self.trackers:
            self._create_new_tracks(detections)
            return self._get_tracks_output()

        matched, unmatched_detections, unmatched_trackers = self._match_detections_to_trackers(detections)

        for track_idx, detection_idx in matched:
            self._update_tracker_hit(track_idx, detections[detection_idx])

        for detection_idx in unmatched_detections:
            self._create_new_tracks([detections[detection_idx]])

        for tracker_idx in unmatched_trackers:
            self._mark_tracker_missed(tracker_idx)

        self._remove_inactive_tracks()

        return self._get_tracks_output()

    def _create_new_tracks(self, detections):
        for detection in detections:
            tracker = self._initiate_tracker(detection)
            self.trackers.append(tracker)

    def _initiate_tracker(self, detection):
        mean, covariance = self._create_kalman_filter_initial_state(detection)
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = self._create_kalman_filter_motion_model()
        kf.H = self._create_kalman_filter_measurement_model()
        kf.x = mean
        kf.P = covariance
        return {
            'kf': kf,
            'time_since_update': 0,
            'hits': 0,
            'hit_streak': 0,
            'age': 0,
            'id': self._next_id,
            'active': False  # Inicialmente inactivo
        }

    def _create_kalman_filter_initial_state(self, detection):
        x1, y1, x2, y2, _ = detection
        mean = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, 0, 0, (x2 - x1), (y2 - y1), 0], dtype=np.float32)
        covariance = np.diag([2, 2, 1e-2, 1e-2, 10, 10, 1e-5])**2
        return mean, covariance

    def _create_kalman_filter_motion_model(self):
        return np.array([
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

    def _create_kalman_filter_measurement_model(self):
        return np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ], dtype=np.float32)

    def _predict_trackers(self):
        for tracker in self.trackers:
            tracker['kf'].predict()
            tracker['age'] += 1

    def _match_detections_to_trackers(self, detections):
        self._predict_trackers()
        if not detections or not self.trackers:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        iou_matrix = self._calculate_iou_matrix(detections)
        matched_indices = self._optimize_iou_assignment(iou_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in [detection_idx for _, detection_idx in matched_indices]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if t not in [tracker_idx for tracker_idx, _ in matched_indices]:
                unmatched_trackers.append(t)

        return matched_indices, unmatched_detections, unmatched_trackers

    def _calculate_iou_matrix(self, detections):
        iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        for i, tracker in enumerate(self.trackers):
            if not tracker['kf'].x[:4].any(): # Para evitar error si kf.x es None o tiene NaN
                continue
            predicted_box = self._get_predicted_box(tracker)
            for j, detection in enumerate(detections):
                detection_box = detection[:4]
                iou_matrix[i, j] = self._iou_bbox(predicted_box, detection_box)
        return iou_matrix

    def _get_predicted_box(self, tracker):
        mean = tracker['kf'].x
        x_center, y_center, _, _, width, height, _ = mean
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return np.array([x1, y1, x2, y2]).astype(np.int32)


    def _iou_bbox(self, box1, box2):
        """Calcula IoU entre dos bounding boxes (formato [x1, y1, x2, y2])."""
        x_inter1 = max(box1[0], box2[0])
        y_inter1 = max(box1[1], box2[1])
        x_inter2 = min(box1[2], box2[2])
        y_inter2 = min(box1[3], box2[3])

        if x_inter2 < x_inter1 or y_inter2 < y_inter1:
            return 0.0

        area_inter = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        area_union = area_box1 + area_box2 - area_inter
        return float(area_inter) / area_union if area_union > 0 else 0.0


    def _optimize_iou_assignment(self, iou_matrix):
        try:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix) # Maximizar IoU es minimizar el negativo del IoU
            matched_indices = []
            for i, row in enumerate(row_ind):
                if iou_matrix[row, col_ind[i]] < self.iou_threshold:
                    continue # Descartar matches con IoU menor al umbral
                matched_indices.append((row, col_ind[i]))
            return matched_indices
        except ValueError: # Si la matriz de IoU está vacía
            return []


    def _update_tracker_hit(self, track_idx, detection):
        tracker = self.trackers[track_idx]
        tracker['time_since_update'] = 0
        tracker['hit_streak'] += 1
        tracker['age'] = 0
        tracker['kf'].update(self._get_measurement_vector(detection))
        if not tracker['active'] and tracker['hits'] >= self.min_hits:
            tracker['active'] = True
        tracker['hits'] += 1


    def _get_measurement_vector(self, detection):
        x1, y1, x2, y2, _ = detection
        return np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, (x2 - x1), (y2 - y1)]).astype(np.float32)


    def _mark_tracker_missed(self, track_idx):
        tracker = self.trackers[track_idx]
        tracker['time_since_update'] += 1
        tracker['hit_streak'] = 0 # Reiniciar hit streak al perder el track


    def _remove_inactive_tracks(self):
        self.trackers = [
            tracker for tracker in self.trackers
            if tracker['time_since_update'] <= self.max_age and (tracker['active'] or tracker['hit_streak'] >= self.min_hits)
        ]
        # Re-indexar IDs si es necesario para mantenerlos consecutivos (opcional para SORT básico)


    def _get_tracks_output(self):
        output_tracks = []
        for tracker in self.trackers:
            if not tracker['active']: # Solo tracks activos
                continue
            mean = tracker['kf'].x[:4]
            x_center, y_center, _, _ = mean
            width = tracker['kf'].x[4]
            height = tracker['kf'].x[5]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            output_tracks.append(np.array([x1, y1, x2, y2, tracker['id']]).astype(np.int32))
        return output_tracks
# --- Fin de la implementación de SORT ---


# Cargar el modelo YOLO y el video
modelo_yolo = YOLO('yolov8n.pt')
video_path = r'C://Users/User/Desktop/video_dron.mp4'  # Reemplaza con la ruta a tu video
cap = cv2.VideoCapture(video_path)
punto_inicial = (100, 200)
punto_final = (500, 200)
color_linea = (0, 255, 0)
grosor_linea = 2

# Inicializar el tracker SORT
sort_tracker = Sort(max_age=2, min_hits=3, iou_threshold=0.3) # Ajusta parámetros según necesidad
contador_carros = 0
linea_y = punto_inicial[1] # Y-coordinate de la línea horizontal para el conteo
carros_cruzados_ids = set() # IDs de tracks de carros ya contados


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resultados = modelo_yolo(frame)
    detecciones_yolo = []
    for resultado in resultados:
        detecciones = resultado.boxes
        for deteccion in detecciones:
            clase_id = int(deteccion.cls[0])
            etiqueta = modelo_yolo.names[clase_id]
            confianza = float(deteccion.conf[0])
            cajas_bb = deteccion.xyxy[0]

            if etiqueta == 'car' and confianza > 0.5:
                x_min, y_min, x_max, y_max = map(int, cajas_bb)
                detecciones_yolo.append([x_min, y_min, x_max, y_max, confianza]) # Formato para SORT


    # Actualizar el tracker SORT con las detecciones YOLO
    tracks = sort_tracker.update(detecciones_yolo)

    cruces_en_este_frame = set() # Para evitar contar un track varias veces en el mismo frame

    # Procesar los tracks para conteo y visualización
    for track in tracks:
        x_min, y_min, x_max, y_max, track_id = map(int, track)
        centro_y_inferior = y_max
        caja_bb_tuple = tuple(track[:4]) # Usar bounding box como ID si es necesario (aunque track_id es mejor)

        # --- Lógica de Cruce de Línea con TRACK_ID ---
        if centro_y_inferior > linea_y and track_id not in carros_cruzados_ids: # Cruzó la línea de arriba a abajo
             contador_carros += 1
             carros_cruzados_ids.add(track_id) # Registrar track_id como contado
             cruces_en_este_frame.add(track_id) # Registrar cruce en este frame (opcional, para debugging)


        # Dibujar bounding box y ID del track
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, f'Carro ID:{track_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # Dibujar la línea de conteo
    cv2.line(frame, punto_inicial, punto_final, color_linea, grosor_linea)

    # Mostrar el contador en el frame
    cv2.putText(frame, f'Carros Contados: {contador_carros}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Deteccion y Conteo de Carros con SORT', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()