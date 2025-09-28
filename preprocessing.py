import kagglehub as kh
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid

# Get the dataset
path = kh.dataset_download("puneet6060/intel-image-classification")
train_dir = f"{path}/seg_train/seg_train"
test_dir = f"{path}/seg_test/seg_test"

# Load train and test data
dataset = ImageFolder(train_dir, transform=transforms.Compose(
    [transforms.Resize((150, 150)), transforms.ToTensor()]))

test_dataset = ImageFolder(test_dir, transform=transforms.Compose(
    [transforms.Resize((150, 150)), transforms.ToTensor()]))


def display_img(img, label):
    print(f"Label: {dataset.classes[label]}")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


def load_data(batch_size=128, val_split=0.2):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Length of Train data: {len(train_dataset)}")
    print(f"Length of Validation data: {len(val_dataset)}")
    print(f"Length of Test data: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size*2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, pin_memory=True)

    return train_loader, val_loader, test_loader


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
    plt.show()