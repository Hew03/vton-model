import os
import json
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# Configuration
IMAGE_SIZE = (256, 256)
CATEGORY_ID = 1  # short sleeve top
SUBMASK_REQUIRED = 3
TEST_VAL_RATIO = 0.15

def load_and_filter_annotations(image_dir, annos_dir):
    """Load annotations and filter for short sleeve tops with 3 submasks."""
    samples = []
    annotation_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]
    
    for filename in tqdm(annotation_files, desc="Processing annotations"):
        with open(os.path.join(annos_dir, filename), 'r') as f:
            data = json.load(f)
            image_path = os.path.join(image_dir, filename.replace('.json', '.jpg'))
            
            for key in data:
                if not key.startswith('item'):
                    continue
                item = data[key]
                if item['category_id'] == CATEGORY_ID and len(item['segmentation']) == SUBMASK_REQUIRED:
                    samples.append({
                        'image_path': image_path,
                        'bbox': item['bounding_box'],
                        'scale': item['scale'],
                        'viewpoint': item['viewpoint'],
                        'zoom_in': item['zoom_in'],
                        'occlusion': item['occlusion'],
                        'style': item['style'],
                        'source': 1 if data['source'] == 'shop' else 0,
                        'segmentation': item['segmentation'],
                        'landmarks': item['landmarks']
                    })
    return samples

def compute_max_seg_length(samples):
    """Calculate maximum segmentation length across all samples."""
    all_seg_lengths = []
    for sample in samples:
        seg_length = sum(len(submask) for submask in sample['segmentation'])
        all_seg_lengths.append(seg_length)
    return max(all_seg_lengths)

def process_sample(sample, max_seg_length):
    """Process a single sample into normalized features/targets with padding."""
    # Load image and get dimensions
    img = cv2.imread(sample['image_path'])
    img_height, img_width = img.shape[:2]
    
    # Normalize bounding box
    x1, y1, x2, y2 = sample['bbox']
    bbox_norm = [
        x1 / img_width, y1 / img_height,
        x2 / img_width, y2 / img_height
    ]
    
    # Crop and resize image patch
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    crop = cv2.resize(crop, IMAGE_SIZE) / 255.0
    
    # Flatten and normalize segmentation
    flattened_seg = []
    for submask in sample['segmentation']:
        flattened_seg.extend(submask)
    flattened_seg = np.array(flattened_seg, dtype=np.float32)
    flattened_seg[::2] /= img_width   # x coordinates
    flattened_seg[1::2] /= img_height  # y coordinates
    
    # Pad segmentation and create mask
    padded_seg = np.zeros(max_seg_length, dtype=np.float32)
    padded_seg[:len(flattened_seg)] = flattened_seg
    seg_mask = np.zeros(max_seg_length, dtype=np.float32)
    seg_mask[:len(flattened_seg)] = 1.0  # 1 for valid points
    
    # Normalize landmarks (fixed length)
    landmarks = np.array(sample['landmarks'], dtype=np.float32)
    landmarks[::3] /= img_width   # x
    landmarks[1::3] /= img_height # y
    
    # One-hot encode categorical features
    categorical_features = [
        sample['scale'] - 1,
        sample['viewpoint'] - 1,
        sample['zoom_in'] - 1,
        sample['occlusion'] - 1,
        sample['style']
    ]
    cat_encoded = tf.keras.utils.to_categorical(categorical_features, num_classes=[3, 3, 3, 3, 10])
    
    return {
        'image_patch': crop,
        'bbox': bbox_norm,
        'source': sample['source'],
        'categorical': np.concatenate(cat_encoded),
        'segmentation': padded_seg,
        'seg_mask': seg_mask,
        'landmarks': landmarks
    }

def create_tfrecord(dataset, output_path):
    """Serialize dataset into TFRecord format with segmentation mask."""
    with tf.io.TFRecordWriter(output_path) as writer:
        for data in dataset:
            # Serialize data
            image_patch = tf.io.serialize_tensor(data['image_patch']).numpy()
            bbox = tf.io.serialize_tensor(data['bbox']).numpy()
            categorical = tf.io.serialize_tensor(data['categorical']).numpy()
            source = tf.io.serialize_tensor([data['source']]).numpy()
            segmentation = tf.io.serialize_tensor(data['segmentation']).numpy()
            seg_mask = tf.io.serialize_tensor(data['seg_mask']).numpy()
            landmarks = tf.io.serialize_tensor(data['landmarks']).numpy()
            
            # Create TFExample
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_patch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_patch])),
                'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
                'categorical': tf.train.Feature(bytes_list=tf.train.BytesList(value=[categorical])),
                'source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source])),
                'segmentation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[segmentation])),
                'seg_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seg_mask])),
                'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[landmarks]))
            }))
            writer.write(example.SerializeToString())

def main():
    # Load and filter data
    samples = load_and_filter_annotations(
        image_dir='data/train/image',
        annos_dir='data/train/annos'
    )
    max_seg_length = compute_max_seg_length(samples)
    
    # Process samples with padding
    processed = [process_sample(s, max_seg_length) for s in tqdm(samples, desc="Processing samples")]
    
    # Split into train/test/val
    train, test_val = train_test_split(processed, test_size=TEST_VAL_RATIO*2, random_state=42)
    test, val = train_test_split(test_val, test_size=0.5, random_state=42)
    
    # Create TFRecords
    os.makedirs('dataset', exist_ok=True)
    create_tfrecord(train, 'dataset/train.tfrecord')
    create_tfrecord(test, 'dataset/test.tfrecord')
    create_tfrecord(val, 'dataset/val.tfrecord')

if __name__ == "__main__":
    main()