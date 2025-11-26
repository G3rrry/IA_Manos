import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# CONFIGURACIÓN
# ==========================
CSV_PATH = "labels.csv"  # Ruta del CSV con nombres de imágenes y etiquetas
IMAGE_DIR = "images"      # Carpeta donde están todas las imágenes
IMG_SIZE = (160, 160)      # Tamaño al que se redimensionarán las imágenes
BATCH_SIZE = 32            # Tamaño de batch para entrenamiento
EPOCHS = 25                # Número de épocas en fase 1
MODEL_PATH = "hand_fingers_model.keras"  # Ruta/nombre del modelo final
NUM_CLASSES = 2            # Número de clases a predecir


def load_and_split_data():
    """
    Carga el CSV que contiene rutas y etiquetas, construye rutas completas
    y divide en entrenamiento/validación manteniendo proporciones de clases.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encontró: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    filenames = df["filename"].astype(str).tolist()  # Asegurar strings
    labels = df["label"].astype(int).tolist()         # Asegurar enteros

    filepaths = [os.path.join(IMAGE_DIR, fn) for fn in filenames]  # Rutas absolutas

    # División estratificada: mantiene proporción de clases
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    return (train_paths, train_labels), (val_paths, val_labels)


def preprocess_image(path, label):
    """
    Lee una imagen desde la ruta, la decodifica, redimensiona y normaliza
    según MobileNetV2 (rango -1 a 1).
    """
    image = tf.io.read_file(path)  # Leer archivo
    image = tf.image.decode_image(image, channels=3, expand_animations=False)  # Decodificar
    image.set_shape((None, None, 3))  # Forzar forma con 3 canales
    image = tf.image.resize(image, IMG_SIZE)  # Redimensionar

    image = preprocess_input(image)  # Normalización específica

    return image, label


def augment(image, label):
    """
    Aumentación- leve en CPU: variaciones de brillo y contraste.
    (Aumentos geométricos se hacen dentro del modelo en GPU).
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def make_dataset(filepaths, labels, is_train=True):
    """
    Crea un pipeline tf.data eficiente con lectura, procesamiento,
    augmentación y batching.
    """
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_train:
        ds = ds.shuffle(len(filepaths), seed=42)  # Mezclar datos
        ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # Optimización
    return ds


def create_model():
    """
    Construye un modelo con MobileNetV2 como base congelada y capas
    de aumento geométrico dentro del modelo.
    """
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    # Aumentación en GPU
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # Cargar MobileNetV2 sin la parte final (include_top=False)
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Congelar en primera fase

    # Pasar datos por base_model sin modificar BatchNorm
    x = base_model(x, training=False)

    # Cabeza de clasificación
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Reduce a vector
    x = tf.keras.layers.Dropout(0.2)(x)              # Regularización
    x = tf.keras.layers.Dense(128, activation="relu")(x)  # Capa intermedia

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Optimización inicial
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",  # Etiquetas como enteros
        metrics=["accuracy"],
    )
    return model


def main():
    print("1. Cargando y dividiendo datos...")
    (train_paths, train_labels), (val_paths, val_labels) = load_and_split_data()
    print(f"   Train: {len(train_paths)} | Validation: {len(val_paths)}")

    print("2. Creando pipelines...")
    train_ds = make_dataset(train_paths, train_labels, is_train=True)
    val_ds = make_dataset(val_paths, val_labels, is_train=False)

    print("3. Construyendo modelo...")
    model = create_model()
    model.summary()

    # Checkpoint guardará el mejor modelo
    checkpoint_path = "best_" + MODEL_PATH

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    print("4. Entrenando (Fase 1: Transfer Learning)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("5. Entrenando (Fase 2: Fine-tuning)...")

    # Buscar la capa MobileNetV2 dentro del modelo
    base_model_layer = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():  # Búsqueda por nombre
            base_model_layer = layer
            print(f"   -> Capa base encontrada: {layer.name}")
            break

    if base_model_layer:
        base_model_layer.trainable = True  # Descongelar

        # Congelar primeras capas y entrenar últimas capas
        fine_tune_at = 100
        for layer in base_model_layer.layers[:fine_tune_at]:
            layer.trainable = False

        # Recompilar con LR muy bajo para no dañar pesos preentrenados
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("   Iniciando Fine-Tuning...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=callbacks,
        )
    else:
        print("   ADVERTENCIA: No se encontró la capa base para fine-tuning.")

    print("Guardando modelo final...")
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
