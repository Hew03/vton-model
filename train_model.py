import os
import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
BUFFER_SIZE = 1000
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_LANDMARKS = 25
NUM_SEGMENTATION_CHANNELS = 3

# Loss weights
SEGMENTATION_LOSS_WEIGHT = 1.0
LANDMARK_LOSS_WEIGHT = 10.0  # Higher weight since landmarks are more precise

# TFRecord parsing functions
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'segmentation': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'lm_mask': tf.io.FixedLenFeature([], tf.string)
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode image
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    
    # Decode segmentation mask
    segmentation = tf.io.parse_tensor(example['segmentation'], out_type=tf.float32)
    segmentation = tf.reshape(segmentation, [*IMAGE_SIZE, NUM_SEGMENTATION_CHANNELS])
    
    # Decode landmarks and landmark mask
    landmarks = tf.io.parse_tensor(example['landmarks'], out_type=tf.float32)
    lm_mask = tf.io.parse_tensor(example['lm_mask'], out_type=tf.float32)
    
    landmarks = tf.reshape(landmarks, [NUM_LANDMARKS * 2])
    lm_mask = tf.reshape(lm_mask, [NUM_LANDMARKS * 2])
    
    return {
        'image': image,
        'segmentation': segmentation,
        'landmarks': landmarks,
        'lm_mask': lm_mask
    }

# Data augmentation functions
def random_flip_horizontal(image, segmentation, landmarks, lm_mask):
    """Randomly flip the image, segmentation, and landmarks horizontally."""
    if tf.random.uniform(()) > 0.5:
        # Flip image and segmentation
        image = tf.image.flip_left_right(image)
        segmentation = tf.image.flip_left_right(segmentation)
        
        # Flip landmarks - we need to adjust x coordinates (even indices)
        x_coords = landmarks[::2]
        y_coords = landmarks[1::2]
        
        # Flip x coordinates (1.0 - x)
        x_coords = 1.0 - x_coords
        
        # Reconstruct landmarks
        landmarks = tf.reshape(tf.stack([x_coords, y_coords], axis=1), [-1])
        
    return image, segmentation, landmarks, lm_mask

def random_brightness_contrast(image, segmentation, landmarks, lm_mask):
    """Apply random brightness and contrast to the image."""
    if tf.random.uniform(()) > 0.5:
        # Apply brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Apply contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Ensure values stay within [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, segmentation, landmarks, lm_mask

def augment_data(data):
    """Apply a series of augmentations to the data."""
    image = data['image']
    segmentation = data['segmentation']
    landmarks = data['landmarks']
    lm_mask = data['lm_mask']
    
    # Apply augmentations
    image, segmentation, landmarks, lm_mask = random_flip_horizontal(image, segmentation, landmarks, lm_mask)
    # Rotation removed due to TF version compatibility issues
    image, segmentation, landmarks, lm_mask = random_brightness_contrast(image, segmentation, landmarks, lm_mask)
    
    return {
        'image': image,
        'segmentation': segmentation,
        'landmarks': landmarks,
        'lm_mask': lm_mask
    }

# Create TF Dataset from TFRecords
def create_dataset(tfrecord_path, augment=False, batch_size=BATCH_SIZE, cache=True):
    """Create dataset from TFRecord files."""
    # Check if the file exists
    if not tf.io.gfile.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
    
    # Create dataset from TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Parse examples
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache data if requested (speeds up training)
    if cache:
        dataset = dataset.cache()
    
    # Apply augmentation during training
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# U-Net blocks for the model
def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same', use_bn=True):
    """Convolutional block with batch normalization."""
    x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    return x

def encoder_block(inputs, filters, kernel_size=3, use_bn=True):
    """Encoder block: Conv -> MaxPool."""
    x = conv_block(inputs, filters, kernel_size, use_bn=use_bn)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(inputs, skip, filters, kernel_size=3, use_bn=True):
    """Decoder block: Upsample -> Concat -> Conv."""
    x = layers.UpSampling2D(size=(2, 2))(inputs)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, kernel_size, use_bn=use_bn)
    return x

# Build the multi-task U-Net model
def build_model(input_shape=(256, 256, 3)):
    """Build a U-Net model with segmentation and landmark heads."""
    inputs = Input(shape=input_shape)
    
    # Encoder
    e1, p1 = encoder_block(inputs, 64)
    e2, p2 = encoder_block(p1, 128)
    e3, p3 = encoder_block(p2, 256)
    e4, p4 = encoder_block(p3, 512)
    
    # Bridge
    b = conv_block(p4, 1024)
    
    # Decoder
    d1 = decoder_block(b, e4, 512)
    d2 = decoder_block(d1, e3, 256)
    d3 = decoder_block(d2, e2, 128)
    d4 = decoder_block(d3, e1, 64)
    
    # Shared features for both tasks
    shared = layers.Conv2D(64, 3, activation='relu', padding='same')(d4)
    
    # Segmentation head (3 channel output)
    seg_head = layers.Conv2D(32, 3, activation='relu', padding='same')(shared)
    seg_output = layers.Conv2D(NUM_SEGMENTATION_CHANNELS, 1, activation='sigmoid', name='segmentation')(seg_head)
    
    # Landmark head
    # Global features for landmarks
    lm_features = layers.GlobalAveragePooling2D()(shared)
    lm_features = layers.Dense(512, activation='relu')(lm_features)
    lm_features = layers.Dropout(0.2)(lm_features)
    lm_features = layers.Dense(256, activation='relu')(lm_features)
    lm_output = layers.Dense(NUM_LANDMARKS * 2, name='landmarks')(lm_features)
    
    # Create model
    model = Model(inputs=inputs, outputs=[seg_output, lm_output])
    
    return model

# Custom loss functions
def masked_mse_loss(lm_mask):
    """MSE loss that's masked by the landmark visibility mask."""
    def loss(y_true, y_pred):
        # Apply mask to both true and predicted landmarks
        masked_true = y_true * lm_mask
        masked_pred = y_pred * lm_mask
        
        # Compute MSE only on visible landmarks
        mse = tf.reduce_sum(tf.square(masked_true - masked_pred)) / (tf.reduce_sum(lm_mask) + tf.keras.backend.epsilon())
        return mse
    return loss

# Custom model wrapper that handles the landmark mask input
class GarmentModel(Model):
    def train_step(self, data):
        # Unpack the data
        x = data['image']
        y = {
            'segmentation': data['segmentation'],
            'landmarks': data['landmarks']
        }
        lm_mask = data['lm_mask']
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            
            # Compute segmentation loss (binary cross entropy)
            seg_loss = self.compiled_loss(
                y['segmentation'], 
                y_pred[0], 
                regularization_losses=self.losses
            )
            
            # Compute landmark loss using masked MSE
            masked_mse = masked_mse_loss(lm_mask)
            lm_loss = masked_mse(y['landmarks'], y_pred[1])
            
            # Total loss
            total_loss = SEGMENTATION_LOSS_WEIGHT * seg_loss + LANDMARK_LOSS_WEIGHT * lm_loss
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y['segmentation'], y_pred[0])
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            'seg_loss': seg_loss,
            'lm_loss': lm_loss,
            'total_loss': total_loss
        })
        return results
    
    def test_step(self, data):
        # Unpack the data
        x = data['image']
        y = {
            'segmentation': data['segmentation'],
            'landmarks': data['landmarks']
        }
        lm_mask = data['lm_mask']
        
        # Forward pass
        y_pred = self(x, training=False)
        
        # Compute segmentation loss
        seg_loss = self.compiled_loss(
            y['segmentation'], 
            y_pred[0], 
            regularization_losses=self.losses
        )
        
        # Compute landmark loss using masked MSE
        masked_mse = masked_mse_loss(lm_mask)
        lm_loss = masked_mse(y['landmarks'], y_pred[1])
        
        # Total loss
        total_loss = SEGMENTATION_LOSS_WEIGHT * seg_loss + LANDMARK_LOSS_WEIGHT * lm_loss
        
        # Update metrics
        self.compiled_metrics.update_state(y['segmentation'], y_pred[0])
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            'seg_loss': seg_loss,
            'lm_loss': lm_loss,
            'total_loss': total_loss
        })
        return results

# Metrics
def iou_metric(y_true, y_pred, threshold=0.5):
    """Calculate IoU for segmentation masks."""
    # Threshold predictions to binary
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true > threshold, tf.float32)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # Calculate IoU
    iou = intersection / (union + tf.keras.backend.epsilon())
    return tf.reduce_mean(iou)

def main():
    # Load dataset counts
    try:
        with open('dataset/samples_count.json', 'r') as f:
            counts = json.load(f)
            print("Dataset counts:", counts)
    except FileNotFoundError:
        print("Warning: samples_count.json not found. Using default steps.")
        counts = {
            'train': 1000,
            'val': 200,
            'test': 200
        }
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset('dataset/train.tfrecord', augment=True, cache=True)
    val_dataset = create_dataset('dataset/val.tfrecord', augment=False, cache=True)
    test_dataset = create_dataset('dataset/test.tfrecord', augment=False, cache=False)
    
    # Build model
    print("Building model...")
    base_model = build_model(input_shape=(*IMAGE_SIZE, 3))
    model = GarmentModel(base_model.inputs, base_model.outputs)
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',  # Used for segmentation loss
        metrics=[iou_metric]
    )
    
    # Print model summary
    model.summary()
    
    # Create output directories
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath='model_checkpoints/best_model.h5',
            save_best_only=True,
            monitor='val_iou_metric',
            mode='max',
            verbose=1
        ),
        # Learning rate scheduling
        ReduceLROnPlateau(
            monitor='val_total_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = counts['train'] // BATCH_SIZE
    validation_steps = max(1, counts['val'] // BATCH_SIZE)
    
    # Train model
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('model_checkpoints/final_model.h5')
    print("Training complete. Final model saved.")
    
    # Save training history
    with open('model_checkpoints/training_history.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serialized_history = {}
        for key, value in history.history.items():
            serialized_history[key] = [float(v) for v in value]
        json.dump(serialized_history, f)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(test_dataset, verbose=1)
    print("Test results:", test_results)

if __name__ == "__main__":
    # Set memory growth to avoid OOM on GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")
    
    # Run main function
    main()