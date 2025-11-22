import cv2
import numpy as np
import tensorflow as tf
import time
import os

# We import the exact preprocessing function used during training
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "hand_fingers_model.keras"  # Updated extension
IMG_SIZE = (160, 160)  # Updated size to match MobileNetV2 input


def preprocess_frame(frame):
    """
    Recibe un frame BGR de OpenCV.
    1. Convierte a RGB.
    2. Redimensiona a 160x160.
    3. Aplica preprocess_input de MobileNet (escala a -1, 1).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(frame_rgb, IMG_SIZE)

    # MobileNetV2 preprocess_input espera valores 0-255 en float
    img_float = img_resized.astype(np.float32)

    # Esto normaliza automáticamente entre -1 y 1
    img_normalized = preprocess_input(img_float)

    return img_normalized


def list_cameras(max_tested=10):
    available = []
    for i in range(max_tested):
        # Usamos CAP_DSHOW en Windows para que sea más rápido escanear
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    import os  # Needed for os.name check in list_cameras

    print("Buscando cámaras...")
    cameras = list_cameras()
    print("Cámaras disponibles indices:", cameras)

    if not cameras:
        print("No se detectaron cámaras.")
        return

    cam_index = int(
        input(f"Ingresa el índice de la cámara a usar ({cameras[0]} por defecto): ")
        or cameras[0]
    )

    cap = cv2.VideoCapture(cam_index)

    print(f"Cargando modelo desde {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Iniciando video. Presiona 'q' para salir.")

    # ============================================
    # Timer variables
    # ============================================
    last_print_time = time.time()
    PRINT_INTERVAL = 2.0  # Reducido a 2s para feedback más rápido
    current_text = "Esperando..."
    # ============================================

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break

        # 1. Preprocesamiento
        img = preprocess_frame(frame)
        img_batch = np.expand_dims(img, axis=0)

        # 2. Predicción
        # predict_on_batch es a veces más rápido para single images en loops
        preds = model.predict_on_batch(img_batch)
        probs = preds[0]  # ej: [0.1, 0.9]
        class_id = np.argmax(probs)
        confidence = probs[class_id]

        # 3. Lógica de clases (Ajusta según tus etiquetas del CSV)
        # Asumimos 0: Closed, 1: Open (o según como entrenaste)
        if class_id == 0:
            label_name = "Mano Cerrada"  # Cambia esto por tu etiqueta real
        else:
            label_name = "Mano Abierta"  # Cambia esto por tu etiqueta real

        display_text = f"{label_name} ({confidence:.2f})"

        # ============================================
        # Logs en consola
        # ============================================
        now = time.time()
        if now - last_print_time >= PRINT_INTERVAL:
            print(
                f"[{time.strftime('%H:%M:%S')}] Pred: {label_name} | Conf: {confidence:.1%}"
            )
            last_print_time = now
        # ============================================

        # 4. Visualización
        # Dibujar un rectángulo de fondo para el texto para mejor lectura
        cv2.rectangle(frame, (30, 10), (400, 60), (0, 0, 0), -1)

        color = (
            (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
        )  # Verde si seguro, Amarillo si duda

        cv2.putText(
            frame,
            display_text,
            (35, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        # Mostrar ventana
        cv2.imshow("Inferencia MobileNetV2", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
