
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class FederatedClient:
    def __init__(self, client_id: int, data_dir: str, model, batch_size: int = 32, img_size: tuple = (224, 224)):
        self.client_id = client_id
        self.data_dir = data_dir
        self.local_model = model
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_data, self.train_samples = self._load_data('train')
        self.val_data, self.val_samples = self._load_data('val')
        self.test_data, self.test_samples = self._load_data('test')
        print(f"Hospital {client_id} Initialized:")
        print(f"   Train: {self.train_samples:,} images")
        print(f"   Val: {self.val_samples:,} images")
        print(f"   Test: {self.test_samples:,} images")

    def _load_data(self, split: str):
        if split == 'train':
            datagen = ImageDataGenerator(
                rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                horizontal_flip=True, zoom_range=0.2, shear_range=0.15, fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()
        data_path = os.path.join(self.data_dir, split)
        data = datagen.flow_from_directory(
            data_path, target_size=self.img_size, batch_size=self.batch_size,
            class_mode='categorical', shuffle=(split == 'train')
        )
        return data, data.n

    def update_local_model(self, global_weights):
        self.local_model.set_weights(global_weights)

    def train_local_model(self, epochs: int = 5, verbose: int = 1):
        print(f"{'─'*60}")
        print(f"Hospital {self.client_id}: Local Training")
        print(f"{'─'*60}")
        history = self.local_model.fit(
            self.train_data, epochs=epochs, validation_data=self.val_data, verbose=verbose,
            steps_per_epoch=max(1, self.train_samples // self.batch_size),
            validation_steps=max(1, self.val_samples // self.batch_size)
        )
        updated_weights = self.local_model.get_weights()
        print(f"Hospital {self.client_id} training complete")
        return history, self.train_samples, updated_weights

    def evaluate_local_model(self):
        results = self.local_model.evaluate(
            self.test_data, steps=max(1, self.test_samples // self.batch_size), verbose=0
        )
        print(f"   Hospital {self.client_id} Test - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        return results
