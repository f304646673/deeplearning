import torchvision

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', download=True)

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))