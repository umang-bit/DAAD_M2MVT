from datasets.daad_dataset import DAADDataset

dataset = DAADDataset()

print(len(dataset))          # Should print 10
sample = dataset[0]

print(sample["views"].keys())  # All views
print(sample["views"]["Front_View"].shape)  # (16, 3, 224, 224)
print(sample["label"])        # Integer
print(sample["clip_id"])      # UUID
