import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, learning_rate, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=96,
                      kernel_size=11,
                      stride=4,
                      padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2),
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      padding=2,
                     groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2),
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      padding=1,
                      groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      padding=1,
                      groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256*6*6,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, 
                      out_features=num_classes)   
        )
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

    def _configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train(self,
            Epochs,
            loss_fn,
            trainloader,
            optimizer=None, 
            accuracy_fn=None,
            device='cpu'):
        self.to(device)
        self.train()

        if optimizer is None:
            optimizer = self._configure_optimizers()

        for epoch in range(Epochs):
            running_loss = 0
            running_acc = 0
            for batch, (X, y) in enumerate(trainloader):
                X, y = X.to(device), y.to(device)
                logits = self(X)
                y_pred = torch.argmax(nn.functional.softmax(logits, dim=1), dim=1) 
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if accuracy_fn is not None:
                    running_acc += accuracy_fn(y_pred, y).item()
            
            avg_loss = running_loss / (batch + 1)
            if accuracy_fn is not None:
                avg_acc = running_acc / (batch + 1)
                print(f"Epoch: {epoch+1}/{Epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            else:
                print(f"Epoch: {epoch+1}/{Epochs}, Loss: {avg_loss:.4f}")
        print("Training completed")

    def test(self,
            testloader,
            accuracy_fn=None,
            device='cpu'):
        self.to(device)
        self.eval()
        with torch.inference_mode():
            for batch, (X, y) in enumerate(testloader):
                X, y = X.to(device), y.to(device)
                y_pred = torch.argmax(nn.Softmax(self(X), dim=1), dim=1)
                acc = accuracy_fn(y, y_pred)
                print(f"Accuracy: {acc}")

    def predict(self, X, device='cpu'):
        self.to(device)
        self.eval()
        with torch.inference_mode():
            y_pred = torch.argmax(nn.Softmax(self(X.to(device)), dim=1), dim=1)
        return y_pred
