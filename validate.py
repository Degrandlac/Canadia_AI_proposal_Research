# from ultralytics import YOLO

# def main():
#     model = YOLO(
#         r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\Baseline\weights\last.pt"
#     )

#     model.val(
#         data=r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\journal-paper-experiment-1-4\data.yaml",
#         split="test",
#         workers=0  # IMPORTANT
#     )

# if __name__ == "__main__":
#     main()
# import os

# def find_empty_annotations(labels_folder):
#     """Find empty annotation files in a folder"""
    
#     empty_files = []
#     valid_files = []
    
#     # Get all .txt files
#     txt_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
#     for txt_file in txt_files:
#         filepath = os.path.join(labels_folder, txt_file)
        
#         # Check if file is empty
#         if os.path.getsize(filepath) == 0:
#             empty_files.append(txt_file)
#         else:
#             # Check if contains only whitespace
#             with open(filepath, 'r') as f:
#                 content = f.read().strip()
#                 if not content:
#                     empty_files.append(txt_file)
#                 else:
#                     valid_files.append(txt_file)
    
#     # Print results
#     print(f"📊 Annotation Analysis")
#     print(f"{'='*50}")
#     print(f"Total files:      {len(txt_files)}")
#     print(f"Valid files:      {len(valid_files)} ({len(valid_files)/len(txt_files)*100:.1f}%)")
#     print(f"Empty/Background: {len(empty_files)} ({len(empty_files)/len(txt_files)*100:.1f}%)")
#     print(f"{'='*50}\n")
    
#     if empty_files:
#         print(f"📄 Empty annotation files ({len(empty_files)}):")
#         for f in empty_files:  # Show first 20
#             print(f"  - {f}")
       
    
#     return empty_files, valid_files

# # Usage
# labels_folder = r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\journal-paper-experiment-1-4\train\labels"
# empty, valid = find_empty_annotations(labels_folder)

# import sys
# sys.path.insert(0, r'C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\Canadia_AI_proposal_Research')

import class_weight
import numpy as np
from collections import Counter
import os

def test_balance(data_yaml_path):
    """Test if weighted sampling balances classes."""
    
    print("\n" + "="*60)
    print("🧪 CLASS BALANCE TEST")
    print("="*60)
    
    # Load dataset info
    import yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get absolute path to training images
    train_path = data.get('train', '')
    
    # Construct proper path
    if not os.path.isabs(train_path):
        base_dir = os.path.dirname(data_yaml_path)
        train_path = os.path.join(base_dir, train_path)
    
    # Normalize path
    train_path = os.path.normpath(train_path)
    
    # Make sure it points to images folder
    if train_path.endswith('labels'):
        train_path = train_path.replace('labels', 'images')
    elif not train_path.endswith('images'):
        # If path doesn't end with images, append it
        if os.path.exists(os.path.join(train_path, 'images')):
            train_path = os.path.join(train_path, 'images')
    
    print(f"\n📁 Using images from: {train_path}")
    
    if not os.path.exists(train_path):
        print(f"\n❌ ERROR: Path does not exist!")
        print(f"   Expected: {train_path}")
        print(f"\n💡 Please check your data.yaml file")
        return
    
    # Test 1: Normal sampling
    print("\n📊 TEST 1: Normal Sampling")
    print("-"*60)
    class_weight.disable_weighted_sampling()
    normal = count_sampled_classes(train_path, samples=500)
    
    # Test 2: Weighted sampling
    print("\n📊 TEST 2: Weighted Sampling")
    print("-"*60)
    class_weight.apply_weighted_sampling()
    weighted = count_sampled_classes(train_path, samples=500)
    
    # Compare
    print("\n" + "="*60)
    print("📈 RESULTS:")
    print("="*60)
    print(f"{'Class':<10} {'Normal %':<15} {'Weighted %':<15} {'Change'}")
    print("-"*60)
    
    normal_total = sum(normal.values())
    weighted_total = sum(weighted.values())
    
    for cls_id in sorted(set(list(normal.keys()) + list(weighted.keys()))):
        n_pct = (normal.get(cls_id, 0) / normal_total) * 100
        w_pct = (weighted.get(cls_id, 0) / weighted_total) * 100
        change = w_pct - n_pct
        
        emoji = "📈" if change > 5 else "📉" if change < -5 else "➡️"
        print(f"Class {cls_id:<4}  {n_pct:5.1f}%         {w_pct:5.1f}%         {emoji} {change:+.1f}%")
    
    # Calculate balance score
    normal_pcts = [normal.get(c, 0) / normal_total * 100 for c in sorted(normal.keys())]
    weighted_pcts = [weighted.get(c, 0) / weighted_total * 100 for c in sorted(weighted.keys())]
    
    normal_std = np.std(normal_pcts)
    weighted_std = np.std(weighted_pcts)
    
    print("\n" + "="*60)
    print(f"📏 Balance Score (lower = more balanced):")
    print(f"   Normal:   {normal_std:.2f}% std deviation")
    print(f"   Weighted: {weighted_std:.2f}% std deviation")
    
    if weighted_std < normal_std:
        improvement = ((normal_std - weighted_std) / normal_std) * 100
        print(f"   ✅ {improvement:.1f}% improvement!")
    
    print("\n✅ Test complete!")
    class_weight.disable_weighted_sampling()


def count_sampled_classes(img_path, samples=500):
    """Sample dataset and count classes."""
    from ultralytics.data.build import YOLODataset
    
    print(f"📂 Loading dataset from: {img_path}")
    
    # Create dataset
    try:
        dataset = YOLODataset(
            img_path=img_path,
            data={'names': {0: '0', 1: '1', 2: '2', 3: '3'}},
        )
    except Exception as e:
        print(f"\n❌ Error creating dataset: {e}")
        print(f"\n💡 Make sure the path exists and contains images")
        raise
    
    print(f"✅ Dataset loaded: {len(dataset)} images")
    
    class_counts = Counter()
    
    print(f"🔄 Sampling {samples} images...")
    
    for i in range(samples):
        idx = np.random.randint(0, len(dataset))
        
        try:
            sample = dataset[idx]
            
            if 'cls' in sample:
                classes = sample['cls'].flatten().astype(int).tolist()
                class_counts.update(classes)
        except Exception as e:
            print(f"  ⚠️ Error sampling index {idx}: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{samples}")
    
    print(f"\n  Results:")
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        pct = (count / sum(class_counts.values())) * 100
        print(f"    Class {cls_id}: {count:3d} samples ({pct:5.1f}%)")
    
    return class_counts


# ============================================
# 🔧 CHANGE THIS PATH TO YOUR data.yaml
# ============================================
if __name__ == "__main__":
    
    data_yaml = r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\journal-paper-experiment-1-4\data.yaml"
    
    test_balance(data_yaml)