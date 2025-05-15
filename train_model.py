import tensorflow as tf
from tensorflow.keras import layers, applications, losses, metrics, optimizers, Model
import numpy as np
import json
import os

# Configuration
BATCH_SIZE = 16  # Adjust based on your VRAM
IMAGE_SIZE = (256, 256)
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 50
LANDMARK_POINTS = 25
LEARNING_RATE = 3e-4

def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'segmentation': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'lm_mask': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode image
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Decode segmentation
    segmentation = tf.io.parse_tensor(example['segmentation'], tf.float32)
    segmentation.set_shape((256, 256, 3))
    
    # Decode landmarks and mask
    landmarks = tf.io.parse_tensor(example['landmarks'], tf.float32)
    landmarks = tf.reshape(landmarks, (LANDMARK_POINTS * 2,))
    
    lm_mask = tf.io.parse_tensor(example['lm_mask'], tf.float32)
    lm_mask = tf.reshape(lm_mask, (LANDMARK_POINTS * 2,))
    
    # Return with a sample weight dictionary
    return image, {
        'segmentation_output': segmentation,
        'landmark_output': landmarks
    }, {
        'segmentation_output': tf.ones((256, 256, 3)),  # use weights of 1.0
        'landmark_output': lm_mask                     # custom mask
    }


def augment_data(image, labels, sample_weight):
    seg = labels['segmentation_output']
    lm = labels['landmark_output']

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)

        seg = tf.stack([seg[...,0], seg[...,2], seg[...,1]], axis=-1)
        labels['segmentation_output'] = seg

        landmarks = tf.reshape(lm, (LANDMARK_POINTS, 2))
        landmarks = tf.stack([1.0 - landmarks[:, 0], landmarks[:, 1]], axis=-1)
        labels['landmark_output'] = tf.reshape(landmarks, (LANDMARK_POINTS * 2,))
    
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)

    # Re-wrap sample_weight as dict
    sample_weight = {
        'segmentation_output': tf.ones_like(labels['segmentation_output']),
        'landmark_output': tf.ones_like(labels['landmark_output']) * sample_weight['landmark_output']
    }

    return image, labels, sample_weight


def create_dataset(filenames, batch_size, augment=False):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def build_model():
    base_model = applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = True
    
    # Segmentation Head
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    segmentation_output = layers.Conv2D(3, 1, padding='same', activation='sigmoid', name='segmentation_output')(x)
    
    # Landmark Head
    y = layers.GlobalAveragePooling2D()(base_model.output)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.Dropout(0.3)(y)
    landmark_output = layers.Dense(LANDMARK_POINTS * 2, activation='sigmoid', name='landmark_output')(y)
    
    model = Model(inputs=base_model.input, outputs=[
        segmentation_output,
        landmark_output
    ])

    return model

class MultiTaskLoss(losses.Loss):
    def __init__(self):
        super().__init__()
        self.seg_loss_fn = losses.BinaryCrossentropy()
        self.lm_loss_fn = losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        seg_true, lm_true = y_true
        seg_pred, lm_pred = y_pred
        seg_loss = self.seg_loss_fn(seg_true, seg_pred)
        lm_loss = self.lm_loss_fn(lm_true, lm_pred)
        return seg_loss + lm_loss

def main():
    # GPU memory optimization
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
    
    # Load datasets
    train_files = tf.data.Dataset.list_files('dataset/train.tfrecord')
    val_files = tf.data.Dataset.list_files('dataset/val.tfrecord')
    
    train_ds = create_dataset(train_files, BATCH_SIZE, augment=True)
    val_ds = create_dataset(val_files, BATCH_SIZE)
    
    # Build and compile model
    model = build_model()
    
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss={
            'segmentation_output': losses.BinaryCrossentropy(),
            'landmark_output': losses.MeanSquaredError()
        },
        metrics={
            'segmentation_output': [metrics.BinaryAccuracy(), metrics.MeanIoU(num_classes=2)],
            'landmark_output': [metrics.MeanAbsoluteError()]
        }
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    for image, labels, weights in train_ds.take(1):
        print("Sample shapes:")
        print("  Image:", image.shape)
        print("  Labels:", {k: v.shape for k, v in labels.items()})
        print("  Weights:", {k: v.shape for k, v in weights.items()})

    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and history
    model.save('final_model.h5')
    
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    main()
