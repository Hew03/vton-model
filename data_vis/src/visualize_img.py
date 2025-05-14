import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_single_file(annos_dir='data/train/annos', images_dir='data/train/image', output_dir='data_vis/results'):
    os.makedirs(output_dir, exist_ok=True)

    all_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]
    
    if not all_files:
        print(f"No annotation files found in {annos_dir}")
        return

    while True:
        selection = input("Enter the number or name of the file to visualize: ").strip()
        
        try:
            file_num = int(selection)
            if 1 <= file_num <= len(all_files):
                filename = all_files[file_num - 1]
                break
            print("Invalid number. Please try again.")
        except ValueError:
            if selection in all_files:
                filename = selection
                break
            print("File not found. Please try again.")

    with open(os.path.join(annos_dir, filename), 'r') as f:
        annotations = json.load(f)
    
    img_path = os.path.join(images_dir, filename.replace('.json', '.jpg'))
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return
        
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    color_palette = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    items = []
    for key in annotations:
        if key.startswith('item'):
            items.append((int(key[4:]), annotations[key]))
    items.sort()
    
    for idx, (item_num, item) in enumerate(items):
        color = color_palette[idx % len(color_palette)]
        cat_name = item.get('category_name', 'Unknown')
        
        if 'segmentation' in item:
            for poly in item['segmentation']:
                if len(poly) >= 6:
                    points = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.polylines(img, [points], True, color, 2)
        
        if 'bounding_box' in item:
            x1, y1, x2, y2 = item['bounding_box']
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"Item{item_num}: {cat_name}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    out_path = os.path.join(output_dir, filename.replace('.json', '_vis.jpg'))
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f"Annotation Visualization: {filename}")
    plt.axis('off')
    plt.show()

    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    visualize_single_file()
