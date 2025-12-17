"""
 Distribution of large, medium, small, and very small objects in different datasets. “Tiny” refers to objects smaller than 16 
×
 16 in size.“Small” refers to objects between 16 
×
 16 and 32 
×
 32 in size.“Medium” refers to objects between 32 
×
 32 and 64 
×
 64 in size.“Large” refers to objects larger than 64 
×
 64 in size.
"""
### compute parasite area
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
DATASET_ROOT = r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\journal-paper-experiment-1-4"
SPLITS = ['train', 'test', 'valid']

# Class names (adjust based on your dataset)
CLASS_NAMES = {
    0: 'PF',  # P. falciparum
    1: 'PM',  # P.malariw
    2: 'PO',  # P. ovale
    3: 'PV',  # P. vivax
    4: 'WBC'   # white blood cell
}

def compute_box_area(bbox, img_width, img_height):
    """
    Compute area in pixels from YOLO normalized bbox
    YOLO format: class x_center y_center width height (all normalized 0-1)
    """
    _, x_center, y_center, width, height = bbox
    
    # Convert to pixel dimensions
    pixel_width = width * img_width
    pixel_height = height * img_height
    
    area = pixel_width * pixel_height
    return area

def analyze_dataset():
    """Analyze all splits and compute area statistics per class"""
    
    results = defaultdict(lambda: {'areas': [], 'count': 0})
    
    for split in SPLITS:
        images_dir = os.path.join(DATASET_ROOT, split, 'images')
        labels_dir = os.path.join(DATASET_ROOT, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"Warning: {labels_dir} not found, skipping...")
            continue
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        for label_file in label_files:
            # Get corresponding image
            img_name = label_file.replace('.txt', '.jpg')
            img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            # Read image dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            
            # Read annotations
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                bbox = parts
                
                area = compute_box_area(bbox, img_width, img_height)
                
                results[class_id]['areas'].append(area)
                results[class_id]['count'] += 1
    
    return results

def print_statistics(results):
    """Print comprehensive statistics"""
    
    print("\n" + "="*70)
    print("PARASITE AREA ANALYSIS")
    print("="*70)
    
    for class_id in sorted(results.keys()):
        areas = results[class_id]['areas']
        
        if not areas:
            continue
        
        class_name = CLASS_NAMES.get(class_id, f'Class_{class_id}')
        
        print(f"\n📊 {class_name} (Class {class_id})")
        print("-" * 50)
        print(f"  Total instances: {len(areas)}")
        print(f"  Mean area:       {np.mean(areas):.2f} px²")
        print(f"  Median area:     {np.median(areas):.2f} px²")
        print(f"  Min area:        {np.min(areas):.2f} px²")
        print(f"  Max area:        {np.max(areas):.2f} px²")
        print(f"  Std deviation:   {np.std(areas):.2f} px²")
        
        # Percentiles
        p25, p75 = np.percentile(areas, [25, 75])
        print(f"  25th percentile: {p25:.2f} px²")
        print(f"  75th percentile: {p75:.2f} px²")
        
        # Equivalent diameter (assuming square)
        mean_side = np.sqrt(np.mean(areas))
        print(f"  Avg side length: {mean_side:.2f} px")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Starting parasite area analysis...")
    results = analyze_dataset()
    print_statistics(results)
    print("\n✅ Analysis complete!")




