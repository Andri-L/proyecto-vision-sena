import cv2

# Lista para almacenar los puntos del polígono
points = []

# Función para capturar clics del mouse
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Si se hace clic izquierdo
        points.append((x, y))  # Guardar las coordenadas
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Dibujar un círculo en el punto
        cv2.imshow("Selecciona el polígono", frame)

# Cargar una imagen o frame desde la cámara
frame = cv2.imread(r'C:/Users/User/Desktop/Screenshot_1.png')  # O usa frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Crear una ventana y asignar la función de clic
cv2.namedWindow("Selecciona el polígono")
cv2.setMouseCallback("Selecciona el polígono", get_coordinates)

while True:
    cv2.imshow("Selecciona el polígono", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
        break

cv2.destroyAllWindows()

# Mostrar las coordenadas seleccionadas
print("Coordenadas del polígono:", points)