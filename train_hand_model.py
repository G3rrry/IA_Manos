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
CSV_PATH = "labels.csv"
IMAGE_DIR = "images"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 25
MODEL_PATH = "hand_fingers_model.keras"  # Changed from .h5 to .keras
NUM_CLASSES = 2


def load_and_split_data():
    """
    Carga el CSV y divide los datos usando sklearn para garantizar
    que train y validación nunca se mezclen.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encontró el archivo: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Asegurarse que sean strings y enteros
    filenames = df["filename"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    filepaths = [os.path.join(IMAGE_DIR, fn) for fn in filenames]

    # Stratified split: Mantiene la proporción de clases en ambos sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    return (train_paths, train_labels), (val_paths, val_labels)


def preprocess_image(path, label):
    """
    Lee la imagen y aplica el pre-procesamiento específico de MobileNetV2 (-1 a 1).
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMG_SIZE)

    # Preprocesamiento específico de MobileNetV2
    image = preprocess_input(image)

    return image, label


def augment(image, label):
    """
    Augmentation ligera en CPU (brillo/contraste).
    La rotación/zoom geométrica se hace en GPU dentro del modelo.
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def make_dataset(filepaths, labels, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_train:
        ds = ds.shuffle(len(filepaths), seed=42)
        ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def create_model():
    """
    Construye el modelo con Transfer Learning y Augmentation layers.
    """
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    # Capas de aumento geométrico (GPU)
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # Base MobileNetV2
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Congelar inicialmente

    # Pasamos el input por el base_model
    # IMPORTANTE: 'training=False' mantiene BatchNorm en modo inferencia
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
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

    # Checkpoint usa .keras ahora
    checkpoint_path = "best_" + MODEL_PATH

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
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

    # --- FIX: Búsqueda robusta de la capa MobileNet ---
    base_model_layer = None
    for layer in model.layers:
        # Buscamos la capa que sea MobileNetV2 (nombre suele contener 'mobilenet')
        if "mobilenet" in layer.name.lower():
            base_model_layer = layer
            print(f"   -> Capa base encontrada: {layer.name}")
            break

    if base_model_layer:
        base_model_layer.trainable = True

        # Congelar las primeras capas, entrenar solo las últimas
        # MobileNetV2 tiene ~155 capas. Entrenamos las últimas 50.
        fine_tune_at = 100
        for layer in base_model_layer.layers[:fine_tune_at]:
            layer.trainable = False

        # Re-compilar con learning rate muy bajo es OBLIGATORIO
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("   Iniciando Fine-Tuning...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,  # Pocas épocas extra
            callbacks=callbacks,
        )
    else:
        print("   ADVERTENCIA: No se pudo encontrar la capa base para fine-tuning.")

    print("Guardando modelo final...")
    model.save(MODEL_PATH)  # Guarda en .keras
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
