from ultralytics_custom.yolo.data.dataset import YOLODataset
import ultralytics_custom.yolo.data.build as build
import numpy as np

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """Initialize the WeightedDataset."""
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)
        
        self.train_mode = "train" in self.prefix
        
        # Calculate class weights automatically
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.agg_func = np.mean
        
        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
        
        # Print statistics
        if self.train_mode:
            print(f"\n📊 Class Distribution:")
            for i, (name, count) in enumerate(zip(self.data["names"].values(), self.counts)):
                print(f"   Class {i} ({name}): {count} instances, weight: {self.class_weights[i]:.2f}")
    
    def count_instances(self):
        """Count the number of instances per class."""
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1
        
        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)  # Avoid division by zero
    
    def calculate_weights(self):
        """Calculate the aggregated weight for each label based on class weights."""
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            
            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue
            
            # Aggregate weights (mean by default)
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights
    
    def calculate_probabilities(self):
        """Calculate and store the sampling probabilities based on the weights."""
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities
    
    def __getitem__(self, index):
        """Return transformed label information based on the sampled index."""
        
        # ✅ FIXED: Use super().__getitem__() instead of get_image_and_label()
        if not self.train_mode:
            # Validation: normal sequential sampling
            return super().__getitem__(index)
        else:
            # Training: weighted random sampling
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return super().__getitem__(index)


def apply_weighted_sampling():
    """Enable weighted class-balanced sampling."""
    build.YOLODataset = YOLOWeightedDataset
    print("✅ Weighted sampling enabled!")


def disable_weighted_sampling():
    """Disable weighted sampling and revert to default."""
    build.YOLODataset = YOLODataset
    print("❌ Weighted sampling disabled")