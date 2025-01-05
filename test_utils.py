import pytest
import os
from PIL import Image
import numpy as np
from utils import walk_through_dir, load_image, random_crop, train, test, eval, accuracy_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# pytest ".test_utils.py" -v

@pytest.fixture
def sample_image():
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    img.save('test_image.png')
    yield 'test_image.png'
    # Cleanup
    os.remove('test_image.png')

def test_walk_through_dir(tmp_path):
    # Create test directory structure
    d = tmp_path / "sub"
    d.mkdir()
    (d / "hello.txt").write_text("content")
    walk_through_dir(tmp_path)

def test_load_image(sample_image):
    size = (50, 50)
    original, img_array = load_image(sample_image, size)
    
    assert isinstance(original, Image.Image)
    assert img_array.shape == (50, 50, 3)
    assert np.all((img_array >= 0) & (img_array <= 1))

def test_load_image_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_image('nonexistent.jpg', (50, 50))

def test_random_crop(sample_image):
    original = Image.open(sample_image)
    crop_size = (50, 50)
    cropped = random_crop(original, crop_size)
    
    assert isinstance(cropped, np.ndarray)
    assert cropped.shape[:2] == crop_size
    import torch.nn as nn

class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)
        
    def forward(self, x):
        return self.layer(x)

@pytest.fixture
def test_data():
    X = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=2)

@pytest.fixture
def test_model():
    return SimpleTestModel()

def test_train_basic(test_data, test_model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01)
    
    initial_params = [p.clone() for p in test_model.parameters()]
    
    train(
        Epochs=2,
        dataloader=test_data,
        Model=test_model,
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    
    # Check parameters were updated
    final_params = [p for p in test_model.parameters()]
    assert not all(torch.equal(i, f) for i, f in zip(initial_params, final_params))
    def test_test_function(test_data, test_model):
        loss_fn = nn.CrossEntropyLoss() 
        optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01)

        # Test basic functionality
        test_model.eval()
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_data):
                y_pred = test_model(X)
                loss = loss_fn(y_pred, y)
                acc = accuracy_fn(y, y_pred.argmax(dim=1))
                
                # Test loss and accuracy calculations
                assert isinstance(loss.item(), float)
                assert isinstance(acc, float)
                assert 0 <= acc <= 100
                
                # Test predictions shape
                assert y_pred.shape[0] == X.shape[0] 
                assert y_pred.shape[1] == 2
                
                # Test device consistency
                assert X.device == y_pred.device
                assert y.device == y_pred.device

        # Test model stays in eval mode
        assert not test_model.training

        # Test with empty batch
        empty_data = DataLoader(TensorDataset(torch.empty(0,4), torch.empty(0)), batch_size=1)
        test(empty_data, test_model, loss_fn, optimizer)

        # Test with device specification
        if torch.cuda.is_available():
            device = torch.device("cuda")
            test_model = test_model.to(device)
            test(test_data, test_model, loss_fn, optimizer, device=device)

def test_eval_function(test_data, test_model):
    loss_fn = nn.CrossEntropyLoss()
    
    results = eval(
        model=test_model,
        data_loader=test_data,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )
    
    assert isinstance(results, dict)
    assert "model_name" in results
    assert "model_loss" in results
    assert "model_acc" in results
    assert results["model_name"] == "SimpleTestModel"
    assert isinstance(results["model_loss"], float)
    assert isinstance(results["model_acc"], float)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_handling(test_data, test_model):
    device = torch.device("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01)
    
    train(
        Epochs=1,
        dataloader=test_data,
        Model=test_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )
    
    assert next(test_model.parameters()).device.type == "cuda"