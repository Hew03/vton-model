import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (256, 256)
NUM_LANDMARKS = 25
NUM_SEGMENTATION_CHANNELS = 3

# Rebuild model architecture
def build_finetuning_model(input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        pooling=None
    )
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    backbone_output = base_model.output
    
    # Segmentation head
    x = tf.keras.layers.Conv2DTranspose(256, (3,3), strides=2, padding='same')(backbone_output)
    x = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=2, padding='same')(x)
    x = tf.keras.layers.UpSampling2D(4)(x)
    seg_output = tf.keras.layers.Conv2D(NUM_SEGMENTATION_CHANNELS, (1,1), activation='sigmoid', name='segmentation')(x)
    
    # Landmark head
    y = tf.keras.layers.GlobalAveragePooling2D()(backbone_output)
    y = tf.keras.layers.Dense(256, activation='relu')(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    lm_output = tf.keras.layers.Dense(NUM_LANDMARKS*2, name='landmarks')(y)
    
    return tf.keras.Model(inputs=base_model.input, outputs=[seg_output, lm_output])

# Load trained weights
model = build_finetuning_model()
model.load_weights('model_checkpoints/final_model.h5')  # Update path if needed

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
    # Load original image to get its dimensions
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = orig_img.shape[:2]
    
    # Resize image for display
    img_resized = cv2.resize(orig_img, IMAGE_SIZE)
    
    plt.figure(figsize=(18, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmentation mask overlay
    plt.subplot(1, 3, 2)
    overlay = np.zeros_like(img_resized)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for i in range(NUM_SEGMENTATION_CHANNELS):
        mask = (seg_mask[:, :, i] > 0.5).astype(np.uint8)
        overlay[mask == 1] = colors[i]
    plt.imshow(cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0))
    plt.title('Segmentation')
    plt.axis('off')
    
    # Landmarks visualization
    plt.subplot(1, 3, 3)
    plt.imshow(img_resized)
    
    # Reshape landmarks to (x,y) pairs
    lm = landmarks.reshape(-1, 2)
    
    # Directly multiply by the visualization dimensions
    # This ensures points are scaled correctly to the displayed image
    x_coords = lm[:, 0] * IMAGE_SIZE[1]
    y_coords = lm[:, 1] * IMAGE_SIZE[0]
    
    # Plot each landmark
    for x, y in zip(x_coords, y_coords):
        plt.scatter(x, y, s=40, c='cyan', edgecolors='black', linewidths=0.5)
    
    plt.title('Landmarks')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return the coordinates for reference
    return list(zip(x_coords, y_coords))

# Add a function to visualize with correctly scaled landmarks on the original image
def visualize_on_original(image_path, seg_mask, landmarks):
    # Load original image to get its dimensions
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = orig_img.shape[:2]
    
    # Create a copy of the original image for visualization
    vis_img = orig_img.copy()
    
    # Reshape landmarks to (x,y) pairs
    lm = landmarks.reshape(-1, 2)
    
    # Scale normalized coordinates to original image dimensions
    x_coords = lm[:, 0] * orig_width
    y_coords = lm[:, 1] * orig_height
    
    # Draw landmarks on the original image
    for x, y in zip(x_coords, y_coords):
        cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.circle(vis_img, (int(x), int(y)), 6, (0, 0, 0), 1)
    
    # Resize segmentation mask to original image size
    resized_seg_mask = np.zeros((orig_height, orig_width, NUM_SEGMENTATION_CHANNELS))
    for i in range(NUM_SEGMENTATION_CHANNELS):
        channel = cv2.resize(seg_mask[:, :, i], (orig_width, orig_height))
        resized_seg_mask[:, :, i] = (channel > 0.5).astype(np.uint8)
    
    # Create segmentation overlay
    overlay = np.zeros_like(vis_img)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for i in range(NUM_SEGMENTATION_CHANNELS):
        mask = resized_seg_mask[:, :, i].astype(np.uint8)
        color_mask = np.zeros_like(vis_img)
        color_mask[mask == 1] = colors[i]
        overlay = np.maximum(overlay, color_mask)
    
    # Combine original image with segmentation overlay
    seg_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
    
    # Display results
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
    
# Example usage
if __name__ == "__main__":
    image_path = 'test_image.jpg'  # Replace with your image path
    seg_mask, landmarks = predict(image_path)
    visualize_results(image_path, seg_mask, landmarks)
    visualize_on_original(image_path, seg_mask, landmarks)