import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_annotations(annos_dir):
    category_dist = defaultdict(int)
    items_per_image = []
    occlusion_dist = defaultdict(int)
    category_pairs = defaultdict(int)
    
    annotation_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]
    
    for filename in tqdm(annotation_files, desc="Processing annotations"):
        with open(os.path.join(annos_dir, filename), 'r') as f:
            data = json.load(f)
            
            items = [v for k, v in data.items() if k.startswith('item')]
            items_per_image.append(len(items))
            
            categories_in_image = []
            
            for item in items:
                category = item['category_name']
                category_id = item['category_id']
                occlusion = item['occlusion']
                
                if category != item['category_name'] or category_id != item['category_id']:
                    print(f"Warning: Inconsistent category in {filename}")
                
                category_dist[category] += 1
                occlusion_dist[occlusion] += 1
                categories_in_image.append(category)
            
            for i in range(len(categories_in_image)):
                for j in range(i+1, len(categories_in_image)):
                    pair = tuple(sorted([categories_in_image[i], categories_in_image[j]]))
                    category_pairs[pair] += 1

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    categories = sorted(category_dist.keys(), key=lambda x: -category_dist[x])
    values = [category_dist[c] for c in categories]
    plt.barh(categories, values)
    plt.title('Category Distribution')
    plt.xlabel('Count')
    
    plt.subplot(2, 2, 2)
    plt.pie(occlusion_dist.values(), labels=occlusion_dist.keys(), autopct='%1.1f%%')
    plt.title('Occlusion Level Distribution')
    
    plt.subplot(2, 2, 3)
    plt.hist(items_per_image, bins=range(0, max(items_per_image)+1))
    plt.title('Items per Image Distribution')
    plt.xlabel('Number of Items')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    sorted_pairs = sorted(category_pairs.items(), key=lambda x: -x[1])[:5]
    pair_labels = [f"{a} + {b}" for (a, b), _ in sorted_pairs]
    pair_values = [count for _, count in sorted_pairs]
    plt.barh(pair_labels, pair_values)
    plt.title('Top 5 Category Co-occurrences')
    
    plt.tight_layout()
    plt.savefig('datavis/results/training_set_analysis.png')
    plt.close()

    print("\nSummary Statistics:")
    print(f"Total items: {sum(category_dist.values())}")
    print(f"Unique categories: {len(category_dist)}")
    print(f"Average items per image: {sum(items_per_image)/len(items_per_image):.2f}")
    print(f"Most common category: {max(category_dist, key=category_dist.get)}")
    print(f"Analysis saved to training_set_analysis.png")

if __name__ == "__main__":
    analyze_annotations('data/train/annos')