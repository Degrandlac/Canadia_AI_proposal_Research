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
import os

def find_empty_annotations(labels_folder):
    """Find empty annotation files in a folder"""
    
    empty_files = []
    valid_files = []
    
    # Get all .txt files
    txt_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        filepath = os.path.join(labels_folder, txt_file)
        
        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            empty_files.append(txt_file)
        else:
            # Check if contains only whitespace
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if not content:
                    empty_files.append(txt_file)
                else:
                    valid_files.append(txt_file)
    
    # Print results
    print(f"📊 Annotation Analysis")
    print(f"{'='*50}")
    print(f"Total files:      {len(txt_files)}")
    print(f"Valid files:      {len(valid_files)} ({len(valid_files)/len(txt_files)*100:.1f}%)")
    print(f"Empty/Background: {len(empty_files)} ({len(empty_files)/len(txt_files)*100:.1f}%)")
    print(f"{'='*50}\n")
    
    if empty_files:
        print(f"📄 Empty annotation files ({len(empty_files)}):")
        for f in empty_files:  # Show first 20
            print(f"  - {f}")
       
    
    return empty_files, valid_files

# Usage
labels_folder = r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\journal-paper-experiment-1-4\train\labels"
empty, valid = find_empty_annotations(labels_folder)