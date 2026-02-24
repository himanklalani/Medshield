
import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import random

def partition_data_for_federated_learning(
    data_dir=r'D:\himank model\brain_tumor_data',
    output_dir=r'D:\himank model\federated_data',
    num_clients=2,
    random_seed=42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    classes = ['Brain_Glioma', 'Brain_Menin', 'Brain_Tumor']

    print(f"\n{'='*60}")
    print(f"PARTITIONING DATA FOR {num_clients} HOSPITALS")
    print(f"{'='*60}")

    # Create client directories
    for client_id in range(num_clients):
        client_dir = os.path.join(output_dir, f'hospital_{client_id + 1}')
        for split in ['train', 'val', 'test']:
            for class_name in classes:
                split_dir = os.path.join(client_dir, split, class_name)
                os.makedirs(split_dir, exist_ok=True)

    # Partition each class
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob(os.path.join(class_path, ext)))
        random.shuffle(image_files)
        print(f"\nClass: {class_name}")
        print(f"  Total images: {len(image_files)}")
        images_per_client = len(image_files) // num_clients

        for client_id in range(num_clients):
            start_idx = client_id * images_per_client
            end_idx = start_idx + images_per_client if client_id < num_clients - 1 else len(image_files)
            client_images = image_files[start_idx:end_idx]
            train_imgs, temp_imgs = train_test_split(client_images, test_size=0.3, random_state=random_seed)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=random_seed)
            client_dir = os.path.join(output_dir, f'hospital_{client_id + 1}')
            for img_path in train_imgs:
                dst = os.path.join(client_dir, 'train', class_name, os.path.basename(img_path))
                shutil.copy2(img_path, dst)
            for img_path in val_imgs:
                dst = os.path.join(client_dir, 'val', class_name, os.path.basename(img_path))
                shutil.copy2(img_path, dst)
            for img_path in test_imgs:
                dst = os.path.join(client_dir, 'test', class_name, os.path.basename(img_path))
                shutil.copy2(img_path, dst)
            print(f"  Hospital {client_id + 1}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    print(f"\n{'='*60}")
    print("DATA PARTITIONING COMPLETED!")
    print(f"{'='*60}")

if __name__ == "__main__":
    partition_data_for_federated_learning()
