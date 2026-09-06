import torch
from torchvision import datasets, transforms

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.ToTensor())

# Get a sample image
sample_image, sample_label = train_dataset[0]

print('=== MNIST Dataset Analysis ===')
print()
print('Image tensor shape:', sample_image.shape)
print('  - Channels:', sample_image.shape[0])
print('  - Height:', sample_image.shape[1])
print('  - Width:', sample_image.shape[2])
print()
print('Value range:')
print('  - Min value:', sample_image.min().item())
print('  - Max value:', sample_image.max().item())
print('  - Values represent: grayscale pixel intensities (0=black, 1=white)')
print()
print('Dataset sizes:')
print('  - Training set:', len(train_dataset), 'images')
print('  - Test set:', len(test_dataset), 'images')
print()
print('Sample label:', sample_label, '(digit shown in the image)')
