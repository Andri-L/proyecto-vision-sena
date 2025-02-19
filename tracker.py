import math 

# Euclidean distance
class Tracker:
    def __init__(self):
        self.center_points = {}  # Almacena los puntos centrales de los bounding boxes
        self.id_count = 1

    def tracker(self, boxes):
        objects = []
        # Obtener el centro de cada bounding box
        for box in boxes:
            x, y, w, h = box
            x_center = x + w // 2
            y_center = y + h // 2

            matched_center = False
            # Verificar si ya se está haciendo seguimiento de este objeto
            for id, point in self.center_points.items():
                distance = math.hypot(x_center - point[0], y_center - point[1])
                if distance < 25:
                    self.center_points[id] = (x_center, y_center)
                    objects.append([x, y, w, h, id])
                    matched_center = True
                    break

            # Si no se encontró coincidencia, se asigna un nuevo ID
            if not matched_center:
                self.center_points[self.id_count] = (x_center, y_center)
                objects.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Actualizar los puntos centrales con los objetos identificados
        new_center_points = {}
        for obj in objects:
            _, _, _, _, object_id = obj
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects