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
NUM_LANDMARKS = 25

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
                        'segmentation': item['segmentation'],
                        'landmarks': item['landmarks']
                    })
    return samples

def process_sample(sample):
    try:
        img = cv2.imread(sample['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        img_height, img_width = img.shape[:2]
        resized_img = cv2.resize(img, IMAGE_SIZE)
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Segmentation mask (3 channels)
        seg_mask = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        for i, submask in enumerate(sample['segmentation']):
            if not submask:
                continue

            poly = np.array(submask, dtype=np.float32).reshape(-1, 2)
            poly[:, 0] *= (IMAGE_SIZE[1] / img_width)
            poly[:, 1] *= (IMAGE_SIZE[0] / img_height)
            
            temp_mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            cv2.fillPoly(temp_mask, [poly.astype(np.int32)], color=1)
            seg_mask[:, :, i] = temp_mask
        
        seg_mask = seg_mask.astype(np.float32)
        
        # Landmarks processing (25 landmarks with visibility)
        landmarks = np.array(sample['landmarks'], dtype=np.float32)
        landmarks = landmarks.reshape(-1, 3)  # Reshape to (25, 3)
        
        # Normalize coordinates
        landmarks[:, 0] /= img_width   # X coordinates
        landmarks[:, 1] /= img_height  # Y coordinates
        
        # Create visibility mask (1 for visible, 0 for invisible)
        lm_mask = (landmarks[:, 2] > 0).astype(np.float32)
        landmarks = landmarks[:, :2].flatten()  # Flatten to 50 values (25x2)
        
        # Encode image
        _, encoded_img = cv2.imencode('.jpg', resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        return {
            'image': encoded_img.tobytes(),
            'segmentation': tf.io.serialize_tensor(tf.convert_to_tensor(seg_mask)).numpy(),
            'landmarks': tf.io.serialize_tensor(tf.convert_to_tensor(landmarks)).numpy(),
            'lm_mask': tf.io.serialize_tensor(tf.convert_to_tensor(lm_mask)).numpy()
        }
    except Exception as e:
        print(f"Skipping sample {sample['image_path']}: {str(e)}")
        return None

def create_tfrecord(samples, output_path):
    valid_count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for sample in tqdm(samples, desc=f"Creating {output_path}"):
            processed = process_sample(sample)
            if processed is None:
                continue
                
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[processed['image']])),
                'segmentation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[processed['segmentation']])),
                'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[processed['landmarks']])),
                'lm_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[processed['lm_mask']]))
            }))
            writer.write(example.SerializeToString())
            valid_count += 1
    print(f"Saved {valid_count}/{len(samples)} valid samples to {output_path}")
    return valid_count

def prepare_tfrecord_dataset():
    samples = load_and_filter_annotations(
        image_dir='data/train/image',
        annos_dir='data/train/annos'
    )
    print(f"Found {len(samples)} valid samples")
    
    if not samples:
        raise ValueError("No valid samples found!")

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

    train_count = create_tfrecord(train_samples, 'dataset/train.tfrecord')
    test_count = create_tfrecord(test_samples, 'dataset/test.tfrecord')
    val_count = create_tfrecord(val_samples, 'dataset/val.tfrecord')
    
    counts = {
        'train': train_count,
        'test': test_count,
        'val': val_count
    }
    with open('dataset/samples_count.json', 'w') as f:
        json.dump(counts, f)

if __name__ == "__main__":
    prepare_tfrecord_dataset()