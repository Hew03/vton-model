import os
import json
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

IMAGE_SIZE = (256, 256)
CATEGORY_ID = 1
SUBMASK_REQUIRED = 3
TEST_VAL_RATIO = 0.15

def load_and_filter_annotations(image_dir, annos_dir):
    samples = []
    annotation_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]
    
    for filename in tqdm(annotation_files, desc="Processing annotations"):
        image_path = os.path.join(image_dir, filename.replace('.json', '.jpg'))
        if not os.path.exists(image_path):
            continue
            
        with open(os.path.join(annos_dir, filename), 'r') as f:
            data = json.load(f)
            for key in data:
                if not key.startswith('item'):
                    continue
                item = data[key]
                if item['category_id'] == CATEGORY_ID and len(item['segmentation']) == SUBMASK_REQUIRED:
                    x1, y1, x2, y2 = item['bounding_box']
                    if x2 <= x1 or y2 <= y1:
                        continue
                    samples.append({
                        'image_path': image_path,
                        'bbox': item['bounding_box'],  # Keep original bbox coordinates
                        'segmentation': item['segmentation'],
                        'landmarks': item['landmarks']
                    })
    return samples

def compute_max_seg_length(samples):
    all_seg_lengths = []
    for sample in samples:
        seg_length = sum(len(submask) for submask in sample['segmentation'])
        all_seg_lengths.append(seg_length)
    return max(all_seg_lengths) if all_seg_lengths else 0

def process_sample(sample, max_seg_length):
    try:
        img = cv2.imread(sample['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        
        img_height, img_width = img.shape[:2]
        
        # Resize full image (no cropping)
        full_img = cv2.resize(img, IMAGE_SIZE) / 255.0
        
        # Convert bbox to normalized [x_center, y_center, width, height] format
        x1, y1, x2, y2 = sample['bbox']
        bbox = np.array([
            ((x1 + x2)/2 / img_width),    # x_center
            ((y1 + y2)/2 / img_height),   # y_center
            (x2 - x1) / img_width,       # width
            (y2 - y1) / img_height       # height
        ], dtype=np.float32)
        
        # Process segmentation
        flattened_seg = np.concatenate(sample['segmentation']).astype(np.float32)
        flattened_seg[::2] /= img_width    # Normalize x coordinates
        flattened_seg[1::2] /= img_height  # Normalize y coordinates
        
        padded_seg = np.zeros(max_seg_length, dtype=np.float32)
        padded_seg[:len(flattened_seg)] = flattened_seg
        seg_mask = np.zeros(max_seg_length, dtype=np.float32)
        seg_mask[:len(flattened_seg)] = 1.0
        
        # Process landmarks
        landmarks = np.array(sample['landmarks'], dtype=np.float32)
        landmarks[::3] /= img_width       # Normalize x coordinates
        landmarks[1::3] /= img_height     # Normalize y coordinates
        
        return {
            'image': full_img,
            'bbox': bbox,
            'segmentation': padded_seg,
            'seg_mask': seg_mask,
            'landmarks': landmarks
        }
    except Exception as e:
        print(f"Skipping sample {sample['image_path']}: {str(e)}")
        return None

def create_tfrecord(samples, output_path, max_seg_length):
    valid_count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for sample in tqdm(samples, desc=f"Creating {output_path}"):
            processed = process_sample(sample, max_seg_length)
            if processed is None:
                continue
                
            image = tf.io.serialize_tensor(processed['image']).numpy()
            bbox = tf.io.serialize_tensor(processed['bbox']).numpy()
            segmentation = tf.io.serialize_tensor(processed['segmentation']).numpy()
            seg_mask = tf.io.serialize_tensor(processed['seg_mask']).numpy()
            landmarks = tf.io.serialize_tensor(processed['landmarks']).numpy()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
                'segmentation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[segmentation])),
                'seg_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seg_mask])),
                'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[landmarks]))
            }))
            writer.write(example.SerializeToString())
            valid_count += 1
    print(f"Saved {valid_count}/{len(samples)} valid samples to {output_path}")

def prepare_tfrecord_dataset():
    samples = load_and_filter_annotations(
        image_dir='data/train/image',
        annos_dir='data/train/annos'
    )
    print(f"Found {len(samples)} valid samples after initial filtering")
    
    if not samples:
        raise ValueError("No valid samples found!")
    
    max_seg_length = compute_max_seg_length(samples)
    print(f"Max segmentation length: {max_seg_length}")
    
    train_samples, test_val_samples = train_test_split(
        samples, 
        test_size=TEST_VAL_RATIO*2, 
        random_state=42
    )
    test_samples, val_samples = train_test_split(
        test_val_samples, 
        test_size=0.5, 
        random_state=42
    )
    
    os.makedirs('dataset', exist_ok=True)
    create_tfrecord(train_samples, 'dataset/train.tfrecord', max_seg_length)
    create_tfrecord(test_samples, 'dataset/test.tfrecord', max_seg_length)
    create_tfrecord(val_samples, 'dataset/val.tfrecord', max_seg_length)

if __name__ == "__main__":
    prepare_tfrecord_dataset()