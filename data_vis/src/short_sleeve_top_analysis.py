import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def analyze_annotations(annos_dir):
    submask_counts = defaultdict(list)
    occlusion_dist_short = defaultdict(int)
    scale_dist = defaultdict(int)
    viewpoint_dist = defaultdict(int)
    zoom_dist = defaultdict(int)
    style_dist = defaultdict(int)
    bbox_widths = []
    bbox_heights = []
    landmark_completeness = []
    landmark_lengths = []

    annotation_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]
    
    for filename in tqdm(annotation_files, desc="Processing annotations"):
        with open(os.path.join(annos_dir, filename), 'r') as f:
            data = json.load(f)
            
            items = [v for k, v in data.items() if k.startswith('item')]
            
            for item in items:
                category = item['category_name']
                if category != "short sleeve top":
                    continue
                
                num_submasks = len(item.get('segmentation', []))
                submask_counts['short sleeve top'].append(num_submasks)
                
                occlusion = item.get('occlusion', 0)
                occlusion_dist_short[occlusion] += 1
                
                scale_dist[item.get('scale', 0)] += 1
                viewpoint_dist[item.get('viewpoint', 0)] += 1
                zoom_dist[item.get('zoom_in', 0)] += 1

                style_dist[item.get('style', 0)] += 1

                bbox = item.get('bounding_box', [0, 0, 0, 0])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox_widths.append(width)
                bbox_heights.append(height)

                landmarks = item.get('landmarks', [])
                total_landmarks = len(landmarks) // 3
                visible_landmarks = sum(1 for i in range(0, len(landmarks), 3) if landmarks[i+2] == 1)
                landmark_completeness.append(visible_landmarks / total_landmarks if total_landmarks > 0 else 0)
                landmark_lengths.append(len(landmarks))

    plt.figure(figsize=(20, 30))

    plt.subplot(5, 2, 1)
    if submask_counts['short sleeve top']:
        max_submask = max(submask_counts['short sleeve top'])
        bins = range(0, max_submask + 2)
    else:
        bins = [0]
    plt.hist(submask_counts['short sleeve top'], bins=bins, edgecolor='black', align='left')
    plt.xticks(range(0, max_submask + 1 if submask_counts['short sleeve top'] else 0))
    plt.title('Segmentation Complexity (Submasks per T-shirt)')
    plt.xlabel('Number of Submasks')
    plt.ylabel('Frequency')
    
    plt.subplot(5, 2, 2)
    plt.pie(
        occlusion_dist_short.values(), 
        labels=[f'Occlusion {k}' for k in occlusion_dist_short.keys()], 
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title('Occlusion Distribution for T-shirts')

    plt.subplot(5, 2, 3)
    plt.bar(scale_dist.keys(), scale_dist.values())
    plt.title('Scale Distribution')
    plt.xlabel('Scale Level')
    plt.ylabel('Count')

    plt.subplot(5, 2, 4)
    plt.bar(viewpoint_dist.keys(), viewpoint_dist.values())
    plt.title('Viewpoint Distribution')
    plt.xlabel('Viewpoint Level')
    plt.ylabel('Count')

    plt.subplot(5, 2, 5)
    plt.bar(zoom_dist.keys(), zoom_dist.values())
    plt.title('Zoom Level Distribution')
    plt.xlabel('Zoom Level')
    plt.ylabel('Count')

    plt.subplot(5, 2, 6)
    plt.scatter(bbox_widths, bbox_heights, alpha=0.5)
    plt.title('Bounding Box Dimensions')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')

    plt.subplot(5, 2, 7)
    plt.hist(landmark_completeness, bins=np.linspace(0, 1, 11), edgecolor='black')
    plt.title('Landmark Visibility Ratio')
    plt.xlabel('Proportion of Visible Landmarks')
    plt.ylabel('Frequency')

    plt.subplot(5, 2, 8)
    plt.bar(style_dist.keys(), style_dist.values())
    plt.title('Style Distribution for T-shirts')
    plt.xlabel('Style ID')
    plt.ylabel('Count')
    
    plt.subplot(5, 2, 9)
    if landmark_lengths:
        bins = range(min(landmark_lengths), max(landmark_lengths) + 2)
        plt.hist(landmark_lengths, bins=bins, edgecolor='black', align='left')
        plt.xticks(range(min(landmark_lengths), max(landmark_lengths) + 1, 5))
    plt.title('Landmark Array Length Distribution')
    plt.xlabel('Length of Landmarks Array')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data_vis/results/short_sleeve_analysis.png')
    plt.close()

    print("\nSummary Statistics:")
    print(f"Average Submasks: {np.mean(submask_counts['short sleeve top']):.2f}" if submask_counts['short sleeve top'] else "No data")
    print(f"Most Common Occlusion: {max(occlusion_dist_short, key=occlusion_dist_short.get) if occlusion_dist_short else 'N/A'}")
    print(f"Dominant Scale: {max(scale_dist, key=scale_dist.get) if scale_dist else 'N/A'}")
    print(f"Dominant Viewpoint: {max(viewpoint_dist, key=viewpoint_dist.get) if viewpoint_dist else 'N/A'}")
    print(f"Average Bounding Box Size: {np.mean(bbox_widths):.1f}x{np.mean(bbox_heights):.1f} pixels" if bbox_widths else "No boxes")
    print(f"Landmark Visibility: {np.mean(landmark_completeness)*100:.1f}%" if landmark_completeness else "N/A")
    print(f"Dominant Style: Style {max(style_dist, key=style_dist.get) if style_dist else 'N/A'}")
    print(f"Average Landmark Array Length: {np.mean(landmark_lengths):.1f}" if landmark_lengths else "N/A")
    print(f"Most Common Landmark Length: {max(set(landmark_lengths), key=landmark_lengths.count) if landmark_lengths else 'N/A'}")

if __name__ == "__main__":
    analyze_annotations('data/train/annos')