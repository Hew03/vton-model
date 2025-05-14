import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics, optimizers
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pyplot as plt
import os
import json

IMG_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 100
SEED = 42
MAX_SEG_LENGTH = 812

def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([], tf.string),
        'segmentation': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'seg_mask': tf.io.FixedLenFeature([], tf.string)
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    
    # Decode and normalize image
    image = tf.io.decode_jpeg(parsed['image'], channels=3)
    image.set_shape([*IMG_SIZE, 3])
    
    # Parse other features
    bbox = tf.io.parse_tensor(parsed['bbox'], tf.float32)
    segmentation = tf.io.parse_tensor(parsed['segmentation'], tf.float32)
    landmarks = tf.io.parse_tensor(parsed['landmarks'], tf.float32)
    seg_mask = tf.io.parse_tensor(parsed['seg_mask'], tf.float32)
    
    bbox.set_shape([4])
    segmentation.set_shape([MAX_SEG_LENGTH])
    landmarks.set_shape([landmarks.shape[0]])
    seg_mask.set_shape([MAX_SEG_LENGTH])
    
    seg_mask = tf.math.round(seg_mask)
    
    return (image, {
        'bbox': bbox,
        'segmentation': segmentation,
        'seg_mask': seg_mask,
        'landmarks': landmarks
    })

def build_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])

def build_triple_head_model(max_seg_length):
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name='image_input')
    
    augmented = build_data_augmentation()(inputs)
    
    base_model = EfficientNetB4(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    x = base_model(augmented)
    
    shared = layers.GlobalAveragePooling2D()(x)
    
    # Bounding Box Head
    b = layers.Dense(256, activation='relu')(shared)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(b)
    
    # Segmentation Head
    s = layers.Dense(512, activation='relu')(shared)
    s = layers.Dense(256, activation='relu')(s)
    seg_output = layers.Dense(max_seg_length, name='segmentation')(s)
    
    # Landmark Head
    l = layers.Dense(512, activation='relu')(shared)
    l = layers.Dropout(0.3)(l)
    lm_output = layers.Dense(75, name='landmarks')(l)
    
    # Segmentation Mask Head
    m = layers.Dense(128, activation='relu')(shared)
    seg_mask_output = layers.Dense(max_seg_length, activation='sigmoid', name='seg_mask')(m)
    
    return models.Model(inputs=inputs, outputs={
        'bbox': bbox_output,
        'segmentation': seg_output,
        'seg_mask': seg_mask_output,
        'landmarks': lm_output
    })

def prepare_dataset(file_pattern, batch_size=32, shuffle=True, repeat=True):
    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=SEED)
    
    dataset = dataset.batch(batch_size)
    
    # Convert batch images to float32 and normalize
    dataset = dataset.map(
        lambda image, targets: (tf.image.convert_image_dtype(image, tf.float32), targets),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if repeat:
        dataset = dataset.repeat()
    
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def train_model():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus[0]}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    with open('dataset/samples_count.json', 'r') as f:
        counts = json.load(f)
    train_samples = counts['train']
    val_samples = counts['val']
    
    train_steps = train_samples // BATCH_SIZE
    val_steps = val_samples // BATCH_SIZE
    
    model = build_triple_head_model(max_seg_length=MAX_SEG_LENGTH)
    model.summary()
    
    optimizer = optimizers.Adam(learning_rate=1e-4)
      
    model.compile(
        optimizer=optimizer,
        loss={
            'bbox': losses.MeanSquaredError(),
            'segmentation': losses.MeanSquaredError(),
            'seg_mask': losses.BinaryCrossentropy(),
            'landmarks': losses.MeanSquaredError()
        },
        loss_weights={
            'bbox': 0.3,
            'segmentation': 0.3,
            'seg_mask': 0.2,
            'landmarks': 0.2
        },
        metrics={
            'bbox': ['mae'],
            'segmentation': ['mae'],
            'seg_mask': ['binary_accuracy'],
            'landmarks': ['mae']
        }
    )
    
    train_ds = prepare_dataset('dataset/train.tfrecord*', batch_size=BATCH_SIZE)
    val_ds = prepare_dataset('dataset/val.tfrecord*', batch_size=BATCH_SIZE, shuffle=False)
    test_ds = prepare_dataset('dataset/test.tfrecord*', batch_size=BATCH_SIZE, shuffle=False, repeat=False)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'output/models/best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_loss',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            verbose=1,
            monitor='val_loss'
        )
    ]
    
    # Create output directories if they don't exist
    os.makedirs('output/models', exist_ok=True)
    
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('output/models/final_model.keras')
    print("Model saved to final_model.keras")
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['bbox_mae'], label='Bbox MAE')
    plt.plot(history.history['landmarks_mae'], label='Landmark MAE')
    plt.title('Metrics')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['segmentation_mae'], label='Seg MAE')
    plt.plot(history.history['seg_mask_binary_accuracy'], label='Seg Mask Acc')
    plt.title('Segmentation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/training_metrics.png')
    plt.close()
    
    test_samples = counts['test']
    test_steps = test_samples // BATCH_SIZE
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_ds, steps=test_steps)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Bbox MAE: {test_results[4]:.4f}")
    print(f"Segmentation MAE: {test_results[5]:.4f}")
    print(f"Seg Mask Acc: {test_results[6]:.4f}")
    print(f"Landmark MAE: {test_results[7]:.4f}")   
    
    return model

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    trained_model = train_model()