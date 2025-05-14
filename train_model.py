import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics, optimizers
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pyplot as plt
import os

# Configuration
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 100
SEED = 42
MAX_SEG_LENGTH = 812  # Matches value from dataset_prep.py

def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([], tf.string),
        'segmentation': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'seg_mask': tf.io.FixedLenFeature([], tf.string)
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    
    image = tf.io.parse_tensor(parsed['image'], tf.float32)
    bbox = tf.io.parse_tensor(parsed['bbox'], tf.float32)
    segmentation = tf.io.parse_tensor(parsed['segmentation'], tf.float32)
    landmarks = tf.io.parse_tensor(parsed['landmarks'], tf.float32)
    seg_mask = tf.io.parse_tensor(parsed['seg_mask'], tf.float32)
    
    image.set_shape([*IMG_SIZE, 3])
    bbox.set_shape([4])
    segmentation.set_shape([MAX_SEG_LENGTH])
    landmarks.set_shape([landmarks.shape[0]])
    seg_mask.set_shape([MAX_SEG_LENGTH])
    
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
    
    # Data augmentation
    augmented = build_data_augmentation()(inputs)
    
    # Base model
    base_model = EfficientNetB4(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    x = base_model(augmented)
    
    # Shared features
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

def prepare_dataset(file_pattern, batch_size=32, shuffle=True):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)
    
    dataset = dataset.repeat()  # Add repeat() to prevent running out of data
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model():
    # Initialize GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus[0]}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Build model
    model = build_triple_head_model(max_seg_length=MAX_SEG_LENGTH)
    model.summary()
    
    # Learning rate schedule
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
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
            'seg_mask': ['accuracy'],
            'landmarks': ['mae']
        }
    )
    
    # Prepare datasets
    train_ds = prepare_dataset('dataset/train.tfrecord*', batch_size=BATCH_SIZE)
    val_ds = prepare_dataset('dataset/val.tfrecord*', batch_size=BATCH_SIZE, shuffle=False)
    test_ds = prepare_dataset('dataset/test.tfrecord*', batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate steps per epoch
    train_steps = sum(1 for _ in tf.data.TFRecordDataset.list_files('dataset/train.tfrecord*')) // BATCH_SIZE
    val_steps = sum(1 for _ in tf.data.TFRecordDataset.list_files('dataset/val.tfrecord*')) // BATCH_SIZE
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',  # Changed to .keras format
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
        ),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('final_model.keras')  # Changed to .keras format
    print("Model saved to final_model.keras")
    
    # Plot training history
    plt.figure(figsize=(18, 6))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Bbox and Landmark metrics
    plt.subplot(1, 3, 2)
    plt.plot(history.history['bbox_mae'], label='Bbox MAE')
    plt.plot(history.history['landmarks_mae'], label='Landmark MAE')
    plt.title('Bbox and Landmark Metrics')
    plt.legend()
    
    # Segmentation metrics
    plt.subplot(1, 3, 3)
    plt.plot(history.history['segmentation_mae'], label='Seg MAE')
    plt.plot(history.history['seg_mask_accuracy'], label='Seg Mask Accuracy')
    plt.title('Segmentation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    print("Training metrics saved to training_metrics.png")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_steps = sum(1 for _ in tf.data.TFRecordDataset.list_files('dataset/test.tfrecord*')) // BATCH_SIZE
    test_results = model.evaluate(test_ds, steps=test_steps)
    print(f"\nTest Results:")
    print(f"Total Loss: {test_results[0]:.4f}")
    print(f"Bbox MAE: {test_results[4]:.4f}")
    print(f"Segmentation MAE: {test_results[5]:.4f}")
    print(f"Seg Mask Accuracy: {test_results[6]:.4f}")
    print(f"Landmark MAE: {test_results[7]:.4f}")
    
    return model

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    trained_model = train_model()