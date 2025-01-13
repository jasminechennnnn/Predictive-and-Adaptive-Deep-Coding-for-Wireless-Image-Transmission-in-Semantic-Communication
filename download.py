import os
import numpy as np
from keras.datasets import cifar10, cifar100

# Create directories if they don't exist
os.makedirs('data/CIFAR10_raw', exist_ok=True)
os.makedirs('data/CIFAR100_raw', exist_ok=True)

# Download and save CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

np.save('data/CIFAR10_raw/x_train.npy', x_train)
np.save('data/CIFAR10_raw/y_train.npy', y_train)
np.save('data/CIFAR10_raw/x_test.npy', x_test)
np.save('data/CIFAR10_raw/y_test.npy', y_test)

print("CIFAR-10 shapes:")
print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_test: {y_test.shape}")

# Download and save CIFAR-100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

np.save('data/CIFAR100_raw/x_train.npy', x_train)
np.save('data/CIFAR100_raw/y_train.npy', y_train)
np.save('data/CIFAR100_raw/x_test.npy', x_test)
np.save('data/CIFAR100_raw/y_test.npy', y_test)

print("\nCIFAR-100 shapes:")
print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_test: {y_test.shape}")