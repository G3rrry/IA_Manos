# train_hand_model.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================
# CONFIGURACIÓN
# ==========================
CSV_PATH = "labels.csv"  # Ruta a tu CSV
IMAGE_DIR = "images"  # Carpeta donde están las imágenes
IMG_SIZE = (128, 128)  # Tamaño al que redimensionamos
BATCH_SIZE = 32
EPOCHS = 25
MODEL_PATH = "hand_fingers_model.keras"

NUM_CLASSES = 2  # dedos 0,1


def load_dataframe():
    """
    Carga el CSV con rutas completas a las imágenes y las etiquetas.
    Ajusta los nombres de columnas aquí si son distintos.
    """
    df = pd.read_csv(CSV_PATH)

    # Ajusta 'filename' y 'label' si tus columnas se llaman distinto
    filenames = df["filename"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    filepaths = [os.path.join(IMAGE_DIR, fn) for fn in filenames]

    return np.array(filepaths), np.array(labels, dtype=np.int32)


def preprocess_image(path, label):
    """
    Lee la imagen desde disco, la decodifica, redimensiona y normaliza.
    Asumimos que las imágenes son JPG/PNG.
    """
    image = tf.io.read_file(path)
    # decode_image soporta JPEG/PNG y fija los canales a 3
    image = tf.image.decode_image(image, channels=3)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment(image, label):
    """
    Data augmentation para mejorar generalización.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label


def make_datasets(filepaths, labels):
    """
    Crea los tf.data.Dataset de entrenamiento y validación.
    """
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.shuffle(len(filepaths), seed=42, reshuffle_each_iteration=True)

    train_size = int(0.8 * len(filepaths))
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    train_ds = (
        train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


def create_model():
    """
    Crea un modelo CNN simple para clasificación de manos.
    """
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # Capa final: 2 clases
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Cargando dataset...")
    filepaths, labels = load_dataframe()
    print(f"Total de imágenes: {len(filepaths)}")

    print("Creando datasets de entrenamiento y validación...")
    train_ds, val_ds = make_datasets(filepaths, labels)

    print("Creando modelo...")
    model = create_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_" + MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("Entrenando...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("Guardando modelo final...")
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
