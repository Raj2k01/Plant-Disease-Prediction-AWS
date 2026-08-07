import tensorflow as tf
from tensorflow.keras import layers


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

BATCH_SIZE = 32

SEED = 42


def load_dataset(dataset_path):

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
    )

    class_names = train_dataset.class_names

    return train_dataset, validation_dataset, class_names

#DAta augmentation

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
    layers.RandomTranslation(0.10, 0.10)
])

normalization = layers.Rescaling(1./255)

AUTOTUNE = tf.data.AUTOTUNE


def prepare_dataset(dataset):

    return dataset.prefetch(buffer_size=AUTOTUNE)