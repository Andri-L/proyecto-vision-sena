import csv
import matplotlib.pyplot as plt
from datetime import datetime
import io

def crear_histograma_distribucion_detecciones_tiempo_csv(csv_file_path):
    """
    Crea un histograma donde el eje X representa el tiempo (timestamps) y el eje Y
    la distribución de la *cantidad* total de detecciones en la 'Zona Roja'
    a lo largo del tiempo.

    Args:
        csv_file_path (str): Ruta al archivo CSV, donde cada fila contiene 'timestamp,detections'.
                          La columna 'detections' debe tener el formato:
                          "Zona Roja: auto=X, camion=Y, ...; Zona Verde: auto=A, camion=B, ..."

    Returns:
        None: Muestra el histograma utilizando matplotlib.pyplot.
    """

    timestamps_histograma = [] # Lista para guardar los timestamps (eje X del histograma)
    cantidades_detecciones = [] # Lista para guardar las cantidades de detecciones (eje Y, aunque matplotlib lo calcula como frecuencia)


    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)

            header = next(csv_reader, None) # Intenta leer el header
            if header:
                if header != ['timestamp', 'detections']: # Si el header no coincide con el esperado
                    csvfile.seek(0) # Volver al inicio si no es el header esperado
                    csv_reader = csv.reader(csvfile) # Re-inicializar sin header asumiendo que no existe

            for row in csv_reader:
                if not row:
                    continue

                try:
                    timestamp_str, detections_str = row
                    timestamp_obj = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S') # Parsear timestamp a objeto datetime

                    total_detecciones_zona_roja = 0 # Inicializar contador de Zona Roja por frame
                    zonas = detections_str.split('; ')

                    for zona_info in zonas:
                        if "Zona Roja:" in zona_info: # Filtrar solo por Zona Roja
                            _, detalles_zona = zona_info.split(': ')
                            vehiculos = detalles_zona.split(', ')
                            for vehiculo_info in vehiculos:
                                if "=" in vehiculo_info:
                                    _, count_str = vehiculo_info.split('=')
                                    try:
                                        total_detecciones_zona_roja += int(count_str)
                                    except ValueError:
                                        print(f"Advertencia: No se pudo convertir conteo a entero para: {vehiculo_info} en timestamp: {timestamp_str}")
                                        continue
                                else:
                                    print(f"Advertencia: Formato inesperado en info vehículo: {vehiculo_info} en timestamp: {timestamp_str}")

                    # Guardar el timestamp *y* la cantidad de detecciones de Zona Roja para este frame
                    timestamps_histograma.append(timestamp_obj) # Eje X: Timestamp
                    cantidades_detecciones.append(total_detecciones_zona_roja) # <---- Eje Y: Cantidad de detecciones

                except ValueError as e:
                    print(f"Error al procesar fila: {row}. Error: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error inesperado al leer CSV: {e}")
        return

    if not timestamps_histograma: # Verificar si se procesaron timestamps antes de graficar
        print("Advertencia: No se encontraron datos válidos para crear el histograma (no hay timestamps).")
        return

    # Crear el histograma donde el eje X es el tiempo y la altura representa la cantidad de detecciones
    plt.figure(figsize=(12, 6)) # Aumentar el tamaño horizontal para mejor visualización de timestamps

    # Usar cantidades_detecciones como 'weights' para que la altura de la barra represente la suma de detecciones
    plt.hist(timestamps_histograma, bins=50, weights=cantidades_detecciones, color='purple', edgecolor='black') # Color morado, pesos añadidos


    plt.title('Histograma de Distribución de Cantidad de Detecciones en Zona Roja a lo largo del Tiempo') # Título actualizado
    plt.xlabel('Timestamp (Tiempo)') # Label de X sigue siendo Tiempo
    plt.ylabel('Cantidad Total de Detecciones en Zona Roja') # Label de Y ahora es "Cantidad de Detecciones"
    plt.grid(axis='y', linestyle='--')
    plt.xticks(rotation=45, ha='right') # Rotar etiquetas del eje X

    plt.tight_layout()
    plt.show()


# Ruta al archivo CSV (asegúrate de que sea la correcta)
csv_file_path = 'C:/Users/User/Desktop/registro_frames.csv' # <----  RUTA AL ARCHIVO CSV AQUÍ

crear_histograma_distribucion_detecciones_tiempo_csv(csv_file_path)