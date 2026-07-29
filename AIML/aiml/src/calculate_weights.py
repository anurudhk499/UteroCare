import os
import torch

dataset_path = "../preprocessed_dataset/train"

classes = os.listdir(dataset_path)

counts = []

for cls in classes:

    class_path = os.path.join(
        dataset_path,
        cls
    )

    count = len(os.listdir(class_path))

    counts.append(count)

total = sum(counts)

weights = []

for count in counts:

    weight = total / count

    weights.append(weight)

weights = torch.tensor(
    weights,
    dtype=torch.float32
)

weights = weights / weights.sum()

print("\nClasses:")
print(classes)

print("\nCounts:")
print(counts)

print("\nNormalized Weights:")
print(weights)