import os
import json
import tensorflow as tf
import gc
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
BUFFER_SIZE = 100
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_LANDMARKS = 25
NUM_SEGMENTATION_CHANNELS = 3

SEGMENTATION_LOSS_WEIGHT = 1.0
LANDMARK_LOSS_WEIGHT = 10.0

def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'segmentation': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'lm_mask': tf.io.FixedLenFeature([], tf.string)
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    
    segmentation = tf.io.parse_tensor(example['segmentation'], out_type=tf.float32)
    segmentation = tf.reshape(segmentation, [*IMAGE_SIZE, NUM_SEGMENTATION_CHANNELS])
    
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

def random_brightness_contrast(image, segmentation, landmarks, lm_mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, segmentation, landmarks, lm_mask

def augment_data(data):
    image = data['image']
    segmentation = data['segmentation']
    landmarks = data['landmarks']
    lm_mask = data['lm_mask']
    
    image, segmentation, landmarks, lm_mask = random_brightness_contrast(image, segmentation, landmarks, lm_mask)
    
    return {
        'image': image,
        'segmentation': segmentation,
        'landmarks': landmarks,
        'lm_mask': lm_mask
    }

def create_dataset(tfrecord_path, augment=False, batch_size=BATCH_SIZE, cache=False):
    if not tf.io.gfile.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if cache:
        dataset = dataset.cache()
    
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    
    return dataset

def build_model(input_shape=(256, 256, 3)):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling=None
    )
    
    for layer in base_model.layers[:100]:
        layer.trainable = False

    input_image = base_model.input

    block_features = []
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.strides == (2, 2):
            if i > 3:
                block_features.append(base_model.layers[i-1].output)
    
    x = base_model.output

    x = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if len(block_features) > 0:
        x = layers.Concatenate()([x, block_features[-1]])

    x = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if len(block_features) > 1:
        x = layers.Concatenate()([x, block_features[-2]])

    x = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if len(block_features) > 2:
        x = layers.Concatenate()([x, block_features[-3]])

    x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    seg_output = layers.Conv2D(
        NUM_SEGMENTATION_CHANNELS, 
        (1, 1), 
        activation='sigmoid', 
        name='segmentation'
    )(x)

    y = layers.Conv2D(64, (3, 3), padding='same')(base_model.output)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Flatten()(y)
    y = layers.Dense(256, activation='relu')(y)
    lm_output = layers.Dense(NUM_LANDMARKS * 2, name='landmarks')(y)
    
    return Model(inputs=input_image, outputs=[seg_output, lm_output])

def binary_crossentropy_loss(y_true, y_pred):
    """Direct implementation of binary crossentropy"""
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def masked_mse_loss(lm_mask):
    def loss(y_true, y_pred):
        masked_true = y_true * lm_mask
        masked_pred = y_pred * lm_mask

        mse = tf.reduce_sum(tf.square(masked_true - masked_pred)) / (tf.reduce_sum(lm_mask) + tf.keras.backend.epsilon())
        return mse
    return loss

def dice_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection) / (union + tf.keras.backend.epsilon())
    return tf.reduce_mean(dice)

def masked_mae(y_true, y_pred, lm_mask):
    masked_true = y_true * lm_mask
    masked_pred = y_pred * lm_mask
    mae = tf.reduce_sum(tf.abs(masked_true - masked_pred)) / (tf.reduce_sum(lm_mask) + tf.keras.backend.epsilon())
    return mae

def iou_metric(y_true, y_pred, threshold=0.5):
    """IoU metric function for model compilation"""
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection

    iou = intersection / (union + tf.keras.backend.epsilon())
    return tf.reduce_mean(iou)

class GarmentModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom metrics
        self.dice_metric = tf.keras.metrics.Mean(name='dice')
        self.mae_metric = tf.keras.metrics.Mean(name='mae')
        self.iou_metric_tracker = tf.keras.metrics.Mean(name='segmentation_iou_metric')
        self.seg_loss_tracker = tf.keras.metrics.Mean(name='seg_loss')
        self.lm_loss_tracker = tf.keras.metrics.Mean(name='lm_loss')
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')

    def train_step(self, data):
        x = data['image']
        y = {
            'segmentation': data['segmentation'],
            'landmarks': data['landmarks']
        }
        lm_mask = data['lm_mask']
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            
            # Calculate seg_loss directly without using compiled_loss
            seg_loss = binary_crossentropy_loss(y['segmentation'], y_pred[0])
            
            masked_mse = masked_mse_loss(lm_mask)
            lm_loss = masked_mse(y['landmarks'], y_pred[1])

            total_loss = SEGMENTATION_LOSS_WEIGHT * seg_loss + LANDMARK_LOSS_WEIGHT * lm_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        dice_value = dice_metric(y['segmentation'], y_pred[0])
        self.dice_metric.update_state(dice_value)

        mae_value = masked_mae(y['landmarks'], y_pred[1], lm_mask)
        self.mae_metric.update_state(mae_value)
        
        iou_value = iou_metric(y['segmentation'], y_pred[0])
        self.iou_metric_tracker.update_state(iou_value)
        
        # Track losses
        self.seg_loss_tracker.update_state(seg_loss)
        self.lm_loss_tracker.update_state(lm_loss)
        self.total_loss_tracker.update_state(total_loss)

        # Build results dict from all metrics
        return {
            'seg_loss': self.seg_loss_tracker.result(),
            'lm_loss': self.lm_loss_tracker.result(),
            'total_loss': self.total_loss_tracker.result(),
            'dice': self.dice_metric.result(),
            'mae': self.mae_metric.result(),
            'segmentation_iou_metric': self.iou_metric_tracker.result()
        }
    
    def test_step(self, data):
        x = data['image']
        y = {
            'segmentation': data['segmentation'],
            'landmarks': data['landmarks']
        }
        lm_mask = data['lm_mask']

        y_pred = self(x, training=False)

        # Calculate seg_loss directly without using compiled_loss
        seg_loss = binary_crossentropy_loss(y['segmentation'], y_pred[0])

        masked_mse = masked_mse_loss(lm_mask)
        lm_loss = masked_mse(y['landmarks'], y_pred[1])

        total_loss = SEGMENTATION_LOSS_WEIGHT * seg_loss + LANDMARK_LOSS_WEIGHT * lm_loss

        # Update metrics
        dice_value = dice_metric(y['segmentation'], y_pred[0])
        self.dice_metric.update_state(dice_value)

        mae_value = masked_mae(y['landmarks'], y_pred[1], lm_mask)
        self.mae_metric.update_state(mae_value)
        
        iou_value = iou_metric(y['segmentation'], y_pred[0])
        self.iou_metric_tracker.update_state(iou_value)
        
        # Track losses
        self.seg_loss_tracker.update_state(seg_loss)
        self.lm_loss_tracker.update_state(lm_loss)
        self.total_loss_tracker.update_state(total_loss)

        # Build results dict from all metrics
        return {
            'seg_loss': self.seg_loss_tracker.result(),
            'lm_loss': self.lm_loss_tracker.result(),
            'total_loss': self.total_loss_tracker.result(),
            'dice': self.dice_metric.result(),
            'mae': self.mae_metric.result(),
            'segmentation_iou_metric': self.iou_metric_tracker.result()
        }

    def reset_metrics(self):
        super().reset_metrics()
        self.dice_metric.reset_state()
        self.mae_metric.reset_state()
        self.iou_metric_tracker.reset_state()
        self.seg_loss_tracker.reset_state()
        self.lm_loss_tracker.reset_state()
        self.total_loss_tracker.reset_state()
        
    @property
    def metrics(self):
        # Define metrics list
        metrics = [
            self.dice_metric,
            self.mae_metric, 
            self.iou_metric_tracker,
            self.seg_loss_tracker,
            self.lm_loss_tracker,
            self.total_loss_tracker
        ]
        return metrics

def main():
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

    print("Creating datasets...")
    train_dataset = create_dataset('dataset/train.tfrecord', augment=True, cache=False)
    val_dataset = create_dataset('dataset/val.tfrecord', augment=False, cache=False)
    test_dataset = create_dataset('dataset/test.tfrecord', augment=False, cache=False)

    gc.collect()

    print("Building model...")
    base_model = build_model(input_shape=(*IMAGE_SIZE, 3))
    model = GarmentModel(base_model.inputs, base_model.outputs)

    print("Compiling model...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
    # Removed loss function and metrics as we're handling those in custom train/test steps

    model.summary()

    os.makedirs('model_checkpoints', exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath='model_checkpoints/best_model.h5',
            save_best_only=True,
            monitor='val_segmentation_iou_metric',
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_total_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    steps_per_epoch = counts['train'] // BATCH_SIZE
    validation_steps = max(1, counts['val'] // BATCH_SIZE)

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

    model.save('model_checkpoints/final_model.h5')
    print("Training complete. Final model saved.")

    with open('model_checkpoints/training_history.json', 'w') as f:
        serialized_history = {}
        for key, value in history.history.items():
            serialized_history[key] = [float(v) for v in value]
        json.dump(serialized_history, f, indent=2)

    print("Evaluating on test set...")
    test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
    print("Test results:", test_results)

    # Save test metrics
    test_metrics = {}
    
    # Process the metrics directly from the return_dict
    for key, value in test_results.items():
        test_metrics[f'test_{key}'] = float(value)
    
    with open('model_checkpoints/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")

    main()