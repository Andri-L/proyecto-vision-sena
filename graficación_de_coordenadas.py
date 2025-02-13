import cv2
import numpy as np

# Definir la resolución del plano
width = 320
height = 240

# Crear un plano en blanco (imagen negra)
img = np.zeros((height, width, 3), dtype=np.uint8)

# Definir las coordenadas DESPLAZADAS de los polígonos
poligono1_coords = np.array([(258, 155), (209, 131), (254, 116), (307, 132)], np.int32) # Polígono Rojo - Desplazado
poligono2_coords = np.array([(81, 240), (318, 130), (319, 176), (220, 240)], np.int32) # Polígono Verde - Desplazado

# Redimensionar los arrays de coordenadas
poligono1_coords = poligono1_coords.reshape((-1, 1, 2))
poligono2_coords = poligono2_coords.reshape((-1, 1, 2))

# Dibujar los polígonos en el plano
# Polígono 1 en color rojo
cv2.polylines(img, [poligono1_coords], isClosed=True, color=(0, 0, 255), thickness=1)
# Polígono 2 en color verde
cv2.polylines(img, [poligono2_coords], isClosed=True, color=(0, 255, 0), thickness=1)

# Mostrar la imagen con los polígonos
cv2.imshow('Poligonos en 320x240 con Poligono Verde en el borde inferior', img)
cv2.waitKey(0)
cv2.destroyAllWindows()