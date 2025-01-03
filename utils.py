# Image Processing
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Model building
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy

from tqdm.auto import tqdm

torch.manual_seed(42)


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def load_image(image_path, size, show=False):
    original_img = Image.open(image_path)
    img = original_img.resize(size)
    img_array = np.array(img)
    # Normalize the image
    img_array = img_array / 255.0
    if show:
        img.show()
    return original_img, img_array

def random_crop(image, size, view=False):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    new_h, new_w = size
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    cropped_image = img_array[top: top + new_h, left: left + new_w]
    if view:
        # View the original image
        plt.figure()
        plt.imshow(cropped_image)
        plt.title("Cropped Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return cropped_image

def rotate_image(image, angle, view=False):
    img_array = np.array(image)
    # img_array = img_array / 255.0
    image_rotated = Image.fromarray(img_array).rotate(angle)
    if view:
        # View the original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis('off')
        # View the modified image
        plt.subplot(1, 2, 2)
        plt.imshow(image_rotated)
        plt.title("Rotated Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    return image_rotated


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train(Epochs: int,
        dataloader: dataloader,
        Model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy_fn=accuracy_fn,
        device=None,
        show_loss_curve=False):
    # For plotting
    loss_list = []
    train_loss, acc = 0, 0
    
    for epoch in tqdm(range(Epochs)):
        print(f"Epoch: {epoch}\n -----")
        Model.train()
        for batch, (X, y) in enumerate(dataloader):
            if device is not None:
                Model = Model.to(device)
                X, y = X.to(device), y.to(device)
            
            y_pred = Model(X)
            loss  = loss_fn(y_pred, y)
            loss_list.append(loss)
            train_loss += loss
            acc = accuracy_fn(y, y_pred.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Test loss: {train_loss:.5f} | Test acc: {acc:.2f}%\n")
    

def test(dataloader: dataloader,
        Model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy_fn=accuracy_fn,
        device=None,):
    loss, acc = 0, 0
    Model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            if device is not None:
                X, y = X.to(device), y.to(device)
            
            y_pred = Model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            print(f"Test loss: {loss:.5f} | Test acc: {acc:.2f}%\n") 

def eval(model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module, 
        accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}


