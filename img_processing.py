import torch
import kagglehub as kh
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from helper import to_device, DeviceDataLoader, get_default_device
from PIL import Image

# Get the dataset
path = kh.dataset_download("puneet6060/intel-image-classification")
train_dir = f"{path}/seg_train/seg_train"
test_dir = f"{path}/seg_test/seg_test"

# Load train and test data
dataset = ImageFolder(train_dir, transform=transforms.Compose(
    [transforms.Resize((150, 150)), transforms.ToTensor()]))

test_dataset = ImageFolder(test_dir, transform=transforms.Compose(
    [transforms.Resize((150, 150)), transforms.ToTensor()]))

device = get_default_device()


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

    train_loader = DataLoader(
        train_dataset, batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size*2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, pin_memory=True)

    # Wrap loaders with DeviceDataLoader
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    return train_loader, val_loader, test_loader


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
    plt.show()


def predict_img_class(img, model):
    """ Predict the class of image and Return Predicted Class"""
    img = to_device(img.unsqueeze(0), device)
    prediction = model(img)
    _, preds = torch.max(prediction, dim=1)
    return dataset.classes[preds[0].item()]


def predict_from_file(path, model):
    # open image file
    img = Image.open(path)
    # convert image to tensor
    img = transforms.Resize((150, 150))(img)
    img = transforms.ToTensor()(img)

    # print image
    img_class = predict_img_class(img, model)
    plt.title(f"Predicted Class : {img_class}")
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig(f'result/{img_class}.png')
    plt.show()

    # prdict image label
    print(f"Predicted Class : {img_class}")
