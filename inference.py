import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

# Constants
IMAGE_SIZE = (256, 256)
NUM_LANDMARKS = 25
NUM_SEGMENTATION_CHANNELS = 3
BATCH_SIZE = 1

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

model = build_model()
model.load_weights('model_checkpoints/final_model.h5')

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

def create_dataset(tfrecord_path, batch_size=BATCH_SIZE):
    if not tf.io.gfile.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    return dataset

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_path):
    img = preprocess_image(image_path)
    seg_pred, lm_pred = model.predict(img)
    return seg_pred[0], lm_pred[0]

def visualize_results(image_path, seg_mask, landmarks):
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(orig_img, IMAGE_SIZE)
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    overlay = np.zeros_like(img_resized)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for i in range(NUM_SEGMENTATION_CHANNELS):
        mask = (seg_mask[:, :, i] > 0.5).astype(np.uint8)
        overlay[mask == 1] = colors[i]
    plt.imshow(cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0))
    plt.title('Segmentation')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_resized)

    lm = landmarks.reshape(-1, 2)

    x_coords = lm[:, 0] * IMAGE_SIZE[1]
    y_coords = lm[:, 1] * IMAGE_SIZE[0]

    for x, y in zip(x_coords, y_coords):
        plt.scatter(x, y, s=40, c='cyan', edgecolors='black', linewidths=0.5)
    
    plt.title('Landmarks')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return list(zip(x_coords, y_coords))

def visualize_on_original(image_path, seg_mask, landmarks):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = orig_img.shape[:2]

    vis_img = orig_img.copy()

    lm = landmarks.reshape(-1, 2)

    x_coords = lm[:, 0] * orig_width
    y_coords = lm[:, 1] * orig_height

    for x, y in zip(x_coords, y_coords):
        cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.circle(vis_img, (int(x), int(y)), 6, (0, 0, 0), 1)

    resized_seg_mask = np.zeros((orig_height, orig_width, NUM_SEGMENTATION_CHANNELS))
    for i in range(NUM_SEGMENTATION_CHANNELS):
        channel = cv2.resize(seg_mask[:, :, i], (orig_width, orig_height))
        resized_seg_mask[:, :, i] = (channel > 0.5).astype(np.uint8)

    overlay = np.zeros_like(vis_img)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for i in range(NUM_SEGMENTATION_CHANNELS):
        mask = resized_seg_mask[:, :, i].astype(np.uint8)
        color_mask = np.zeros_like(vis_img)
        color_mask[mask == 1] = colors[i]
        overlay = np.maximum(overlay, color_mask)

    seg_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(seg_img)
    plt.title('Segmentation on Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(vis_img)
    plt.title('Landmarks on Original')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return list(zip(x_coords, y_coords))

def visualize_prediction_vs_ground_truth(image, gt_seg, gt_landmarks, gt_lm_mask=None):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(gt_seg, tf.Tensor):
        gt_seg = gt_seg.numpy()
    if isinstance(gt_landmarks, tf.Tensor):
        gt_landmarks = gt_landmarks.numpy()
    if isinstance(gt_lm_mask, tf.Tensor) and gt_lm_mask is not None:
        gt_lm_mask = gt_lm_mask.numpy()
    
    img_batch = np.expand_dims(image, axis=0) if image.shape[0] != 1 else image
    pred_seg, pred_landmarks = model.predict(img_batch)
    
    pred_seg = pred_seg[0]
    pred_landmarks = pred_landmarks[0]
    
    vis_img = (image * 255).astype(np.uint8)
    
    plt.figure(figsize=(20, 15))

    plt.subplot(3, 3, 1)
    plt.imshow(vis_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    gt_overlay = np.zeros_like(vis_img)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for i in range(NUM_SEGMENTATION_CHANNELS):
        mask = (gt_seg[:, :, i] > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(vis_img)
        colored_mask[mask == 1] = colors[i]
        gt_overlay = np.maximum(gt_overlay, colored_mask)
    
    plt.imshow(cv2.addWeighted(vis_img, 0.7, gt_overlay, 0.3, 0))
    plt.title('Ground Truth Segmentation')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    pred_overlay = np.zeros_like(vis_img)
    for i in range(NUM_SEGMENTATION_CHANNELS):
        mask = (pred_seg[:, :, i] > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(vis_img)
        colored_mask[mask == 1] = colors[i]
        pred_overlay = np.maximum(pred_overlay, colored_mask)
    
    plt.imshow(cv2.addWeighted(vis_img, 0.7, pred_overlay, 0.3, 0))
    plt.title('Predicted Segmentation')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(vis_img)
    
    gt_lm = gt_landmarks.reshape(-1, 2)
    gt_x_coords = gt_lm[:, 0] * IMAGE_SIZE[1]
    gt_y_coords = gt_lm[:, 1] * IMAGE_SIZE[0]

    if gt_lm_mask is not None:
        gt_lm_mask = gt_lm_mask.reshape(-1, 2)
        mask_x = gt_lm_mask[:, 0] > 0
        mask_y = gt_lm_mask[:, 1] > 0
        mask = mask_x & mask_y

        gt_x_coords = gt_x_coords[mask]
        gt_y_coords = gt_y_coords[mask]
    
    for x, y in zip(gt_x_coords, gt_y_coords):
        plt.scatter(x, y, s=40, c='green', edgecolors='black', linewidths=0.5)
    
    plt.title('Ground Truth Landmarks')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(vis_img)
    
    pred_lm = pred_landmarks.reshape(-1, 2)
    pred_x_coords = pred_lm[:, 0] * IMAGE_SIZE[1]
    pred_y_coords = pred_lm[:, 1] * IMAGE_SIZE[0]

    if gt_lm_mask is not None:
        gt_lm_mask = gt_lm_mask.reshape(-1, 2)
        mask_x = gt_lm_mask[:, 0] > 0
        mask_y = gt_lm_mask[:, 1] > 0
        mask = mask_x & mask_y
        
        # Filter coordinates where mask is True
        pred_x_coords = pred_x_coords[mask]
        pred_y_coords = pred_y_coords[mask]
    
    for x, y in zip(pred_x_coords, pred_y_coords):
        plt.scatter(x, y, s=40, c='cyan', edgecolors='black', linewidths=0.5)
    
    plt.title('Predicted Landmarks')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    combined_img = vis_img.copy()

    for i in range(NUM_SEGMENTATION_CHANNELS):
        gt_mask = (gt_seg[:, :, i] > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined_img, contours, -1, (0, 255, 0), 2)

    for i in range(NUM_SEGMENTATION_CHANNELS):
        pred_mask = (pred_seg[:, :, i] > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined_img, contours, -1, (255, 0, 0), 1)
    
    plt.imshow(combined_img)
    plt.title('Segmentation Comparison (Green=GT, Red=Pred)')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(vis_img)

    for x, y in zip(gt_x_coords, gt_y_coords):
        plt.scatter(x, y, s=40, c='green', edgecolors='black', linewidths=0.5)

    for x, y in zip(pred_x_coords, pred_y_coords):
        plt.scatter(x, y, s=20, c='red', edgecolors='black', linewidths=0.5)
    
    plt.title('Landmark Comparison (Green=GT, Red=Pred)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    return

def visualize_from_dataset(dataset_path, num_samples=5):
    dataset = create_dataset(dataset_path)
    dataset = dataset.skip(1)
    
    metrics_list = []
    
    for i, sample_batch in enumerate(dataset):
        if i >= num_samples:
            break
            
        sample = {key: value[0] for key, value in sample_batch.items()}
        
        print(f"\n--- Sample {i+1} ---")
        visualize_prediction_vs_ground_truth(
            sample['image'], 
            sample['segmentation'], 
            sample['landmarks'],
            sample['lm_mask']
        )
    
    if metrics_list:
        avg_iou = np.mean([m['avg_iou'] for m in metrics_list])
        avg_landmark_mse = np.mean([m['landmark_mse'] for m in metrics_list])
        
        print("\n--- Overall Metrics ---")
        print(f"Average IoU across samples: {avg_iou:.4f}")
        print(f"Average Landmark MSE across samples: {avg_landmark_mse:.6f}")
        
        return metrics_list
    
    return None

if __name__ == "__main__":
    print("1. Processing a single image")
    image_path = 'test_image.jpg'
    seg_mask, landmarks = predict(image_path)
    visualize_results(image_path, seg_mask, landmarks)
    visualize_on_original(image_path, seg_mask, landmarks)
    
    print("\n2. Visualizing samples from test dataset with ground truth")
    visualize_from_dataset('dataset/test.tfrecord', num_samples=2)
