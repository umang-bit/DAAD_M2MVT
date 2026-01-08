from datasets.daad_dataset import DAADDataset

dataset = DAADDataset()

sample = dataset[0]

print("Gaze shape:", sample["gaze"].shape)
print(sample["gaze"][:5])
