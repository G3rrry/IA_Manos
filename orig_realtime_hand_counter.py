import cv2
import numpy as np
import tensorflow as tf
import time

MODEL_PATH = "hand_fingers_model.keras"
IMG_SIZE = (128, 128)


def preprocess_frame(frame):
    """
    Recibe un frame BGR de OpenCV, lo convierte a RGB,
    lo redimensiona y normaliza a [0,1].
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(frame_rgb, IMG_SIZE)
    img_resized = img_resized.astype("float32") / 255.0
    return img_resized


def list_cameras(max_tested=10):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available


def main():
    print("Cámaras disponibles:", list_cameras())

    cam_index = int(input("Ingresa el índice de la cámara a usar: "))
    cap = cv2.VideoCapture(cam_index)

    print("Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado.")

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    # ============================================
    # Timer para impresión cada 5 segundos
    # ============================================
    last_print_time = time.time()
    PRINT_INTERVAL = 5.0  # segundos
    # ============================================

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # Preprocesar frame para el modelo
        img = preprocess_frame(frame)
        img_batch = np.expand_dims(img, axis=0)

        # Predicción
        preds = model.predict(img_batch, verbose=0)
        probs = preds[0]  # ej: [0.23, 0.77]
        class_id = int(np.argmax(probs))

        # Obtener texto legible
        if class_id == 0:
            text = "Mano cerrada"
        else:
            text = "Mano abierta"

        # ============================================
        # Imprimir probabilidad cada 5 segundos
        # ============================================
        now = time.time()
        if now - last_print_time >= PRINT_INTERVAL:
            print(f"[{time.strftime('%H:%M:%S')}] Probabilidades: {probs}")
            print(f" → Predicción: {text}")
            last_print_time = now
        # ============================================

        # Dibujar texto en pantalla
        cv2.putText(
            frame,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Mostrar ventana
        cv2.imshow("Contador de dedos", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
