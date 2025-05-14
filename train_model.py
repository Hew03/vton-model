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

def parse_tfrecord(example):
    """Parse TFRecord entries into properly structured tensors"""
    feature_description = {
        'image_patch': tf.io.FixedLenFeature([], tf.string),
        'segmentation': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'seg_mask': tf.io.FixedLenFeature([], tf.string)
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    
    # Decode tensors
    image = tf.io.parse_tensor(parsed['image_patch'], tf.float32)
    segmentation = tf.io.parse_tensor(parsed['segmentation'], tf.float32)
    landmarks = tf.io.parse_tensor(parsed['landmarks'], tf.float32)
    
    # Reshape tensors
    image = tf.reshape(image, [*IMG_SIZE, 3])
    segmentation = tf.reshape(segmentation, [*IMG_SIZE, 1])
    landmarks = tf.reshape(landmarks, [8])  # Assuming 4 landmarks (x,y)*4
    
    # Return inputs and outputs in proper format
    return (image, {'segmentation': segmentation, 'landmarks': landmarks})

def build_data_augmentation():
    """Create data augmentation layer"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])

def build_dual_head_model():
    """Build model with EfficientNet backbone and dual outputs"""
    # Input layer
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name='image_input')
    
    # Data augmentation
    augmented = build_data_augmentation()(inputs)
    
    # Preprocessing
    x = layers.Rescaling(1./255)(augmented)
    
    # Base model
    base_model = EfficientNetB4(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    x = base_model(x)
    
    # Segmentation Head - Now correctly outputs 256x256
    s = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    s = layers.UpSampling2D(16)(s)  # 8*16=128
    s = layers.Conv2D(128, 3, padding='same', activation='relu')(s)
    s = layers.UpSampling2D(2)(s)  # 128*2=256
    s = layers.Conv2D(64, 3, padding='same', activation='relu')(s)
    seg_output = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='segmentation')(s)
    
    # Landmark Head
    l = layers.GlobalAveragePooling2D()(x)
    l = layers.Dense(512, activation='relu')(l)
    l = layers.Dropout(0.3)(l)
    lm_output = layers.Dense(8, name='landmarks')(l)
    
    return models.Model(inputs=inputs, outputs=[seg_output, lm_output])

def prepare_dataset(file_pattern, batch_size=32, shuffle=True):
    """Create optimized dataset pipeline"""
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model():
    """Complete training pipeline"""
    # Initialize GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus[0]}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Build model
    model = build_dual_head_model()
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={
            'segmentation': losses.BinaryCrossentropy(),
            'landmarks': losses.MeanSquaredError()
        },
        loss_weights=[0.7, 0.3],
        metrics={
            'segmentation': ['accuracy', metrics.MeanIoU(2)],
            'landmarks': ['mae']
        }
    )
    
    # Prepare datasets
    train_ds = prepare_dataset('dataset/train.tfrecord*', batch_size=BATCH_SIZE)
    val_ds = prepare_dataset('dataset/val.tfrecord*', batch_size=BATCH_SIZE, shuffle=False)
    test_ds = prepare_dataset('dataset/test.tfrecord*', batch_size=BATCH_SIZE, shuffle=False)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
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
        callbacks=callbacks
    )
    
    # Save final model
    model.save('final_model.h5')
    print("Model saved to final_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['segmentation_accuracy'], label='Seg Accuracy')
    plt.plot(history.history['landmarks_mae'], label='Landmark MAE')
    plt.title('Training Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    print("Training metrics saved to training_metrics.png")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_ds)
    print(f"\nTest Results:")
    print(f"Segmentation Accuracy: {test_results[3]:.4f}")
    print(f"Segmentation IoU: {test_results[4]:.4f}")
    print(f"Landmark MAE: {test_results[5]:.4f}")
    
    return model

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Run training
    trained_model = train_model()