import os
import json
import numpy as np
import tensorflow as tf
from skimage.draw import polygon
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
IMG_SIZE = (512, 512)
MAX_PARTS = 6
MAX_LANDMARKS = 20
CATEGORY_ID = 1
DATA_ROOT = 'data'
OUTPUT_DIR = 'processed_data'

def parse_annotations(data_dir):
    """Parse annotations and return paths + metadata"""
    samples = []
    ann_dir = os.path.join(data_dir, 'annos')
    
    for fname in tqdm(os.listdir(ann_dir)):
        if not fname.endswith('.json'):
            continue
            
        with open(os.path.join(ann_dir, fname), 'r') as f:
            data = json.load(f)
            img_id = fname.split('.')[0]
            tshirt_items = [
                item for key, item in data.items()
                if key.startswith('item') and item['category_id'] == CATEGORY_ID
            ]
            
            if tshirt_items:
                # Store only the necessary metadata
                samples.append({
                    'img_path': os.path.join(data_dir, 'image', f"{img_id}.jpg"),
                    'seg_data': [item['segmentation'] for item in tshirt_items],
                    'landmark_data': [item['landmarks'] for item in tshirt_items],
                    'img_size': [data['item1']['bounding_box'][2],  # Width
                                 data['item1']['bounding_box'][3]] # Height
                })
    
    return samples

def create_part_masks(segmentation, img_width, img_height):
    """Create multi-channel mask for t-shirt parts"""
    mask = np.zeros((img_height, img_width, MAX_PARTS), dtype=np.float32)
    for part_idx, poly_group in enumerate(segmentation[:MAX_PARTS]):
        for poly in poly_group:
            if len(poly) >= 6:
                rr, cc = polygon(np.array(poly[1::2]), np.array(poly[0::2]))
                valid = (rr < img_height) & (cc < img_width)
                mask[rr[valid], cc[valid], part_idx] = 1.0
    return mask

def process_landmarks(landmarks, img_width, img_height):
    """Normalize and pad landmarks with visibility mask"""
    coords = []
    vis_mask = []
    
    for i in range(0, len(landmarks), 3):
        x, y, v = landmarks[i], landmarks[i+1], landmarks[i+2]
        if v > 0 and x < img_width and y < img_height:
            coords.extend([x/img_width, y/img_height])
            vis_mask.append(1.0)
        else:
            coords.extend([0.0, 0.0])
            vis_mask.append(0.0)
    
    # Pad to MAX_LANDMARKS*2
    pad_len = MAX_LANDMARKS*2 - len(coords)
    return (
        np.array(coords + [0.0]*pad_len, dtype=np.float32),
        np.array(vis_mask + [0.0]*pad_len, dtype=np.float32)
    )


def process_sample(sample):
    """Process single sample"""
    # Load image
    image = tf.image.decode_jpeg(tf.io.read_file(sample['img_path']), channels=3)
    orig_h, orig_w = sample['img_size']
    
    # Process masks
    combined_mask = np.zeros((orig_h, orig_w, MAX_PARTS), dtype=np.float32)
    all_landmarks = []
    all_vis_masks = []
    
    for seg, landmarks in zip(sample['seg_data'], sample['landmark_data']):
        # Create part mask
        mask = create_part_masks(seg, orig_w, orig_h)
        combined_mask = np.maximum(combined_mask, mask)
        
        # Process landmarks
        coords, vis_mask = process_landmarks(landmarks, orig_w, orig_h)
        all_landmarks.append(coords)
        all_vis_masks.append(vis_mask)
    
    # Resize and normalize
    image = tf.image.resize(image, IMG_SIZE) / 255.0
    mask = tf.image.resize(combined_mask, IMG_SIZE)
    landmarks = tf.concat(all_landmarks, axis=0) if all_landmarks else tf.zeros((MAX_LANDMARKS*2,))
    vis_masks = tf.concat(all_vis_masks, axis=0) if all_vis_masks else tf.zeros((MAX_LANDMARKS*2,))
    
    return image, (mask, landmarks, vis_masks)

# Create dataset from paths and metadata
all_samples = parse_annotations(os.path.join(DATA_ROOT, 'train'))
train_val, test = train_test_split(all_samples, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.2, random_state=42)

# Convert to TensorFlow dataset
def create_dataset(samples):
    return tf.data.Dataset.from_tensor_slices({
        'img_path': [s['img_path'] for s in samples],
        'seg_data': [s['seg_data'] for s in samples],
        'landmark_data': [s['landmark_data'] for s in samples],
        'img_size': [s['img_size'] for s in samples]
    }).map(
        lambda x: tf.py_function(
            process_sample, [x],
            (tf.float32, (tf.float32, tf.float32, tf.float32))
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

# Create and save datasets
train_ds = create_dataset(train).shuffle(1000).batch(8).prefetch(2)
val_ds = create_dataset(val).batch(8).prefetch(2)
test_ds = create_dataset(test).batch(8).prefetch(2)

# Save datasets
os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.data.Dataset.save(train_ds, os.path.join(OUTPUT_DIR, 'train'))
tf.data.Dataset.save(val_ds, os.path.join(OUTPUT_DIR, 'val'))
tf.data.Dataset.save(test_ds, os.path.join(OUTPUT_DIR, 'test'))
