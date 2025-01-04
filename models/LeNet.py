import torch
from torch import nn


class LeNet(nn.Module):
        def __init__(self, learning_rate, num_classes):
                super().__init__()
                self.conv_block = nn.Sequential(
                        nn.Conv2d(in_channels=1, 
                                out_channels=6, 
                                kernel_size=5, 
                                stride=1, 
                                padding=0),
                        nn.Tanh(),
                        nn.MaxPool2d(kernel_size=2, 
                        stride=2),
                        nn.Conv2d(in_channels=6,
                                out_channels=16,
                                kernel_size=5,
                                stride=1,
                                padding=0),
                        nn.Tanh(),
                        nn.MaxPool2d(kernel_size=2, 
                        stride=2)
                )
                self.fc_block = nn.Sequential(
                        nn.Linear(in_features=16*5*5, 
                        out_features=120),
                        nn.ReLU(),
                        nn.Linear(in_features=120, 
                        out_features=84),
                        nn.ReLU(),
                        nn.Linear(in_features=84, 
                        out_features=num_classes)
                )

        def forward(self, x):
                x = self.conv_block(x)
                x = torch.flatten(x, start_dim=1)  # flattens all dimensions except batch (dim 0)
                x = self.fc_block(x)
                return x
        
        def _configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                return optimizer
        
        def train(self, 
                trainloader,
                Epochs, 
                loss_fn, 
                optimizer=None, 
                accuracy_fn=None, 
                device='cpu'):
                self.to(device)
                self.train()
                
                if optimizer is None:
                        optimizer = self._configure_optimizers()

                for epoch in range(Epochs):
                        for batch, (X, y) in enumerate(trainloader):
                                X, y = X.to(device), y.to(device)
                                logits = self(X)
                                y_hat = torch.argmax(nn.functional.softmax(logits, dim=1), dim=1)
                                loss = loss_fn(logits, y) 
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                
                                if accuracy_fn is not None:
                                        accuracy = accuracy_fn(y_hat, y)
                                        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")
                                else:
                                        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        
                
        def eval(self, testloader, accuracy_fn=None):
                self.eval()
                return {"Name": self.__class__.__name__,
                        "Accuracy": acc}

        def predict(self, X, device='cpu'):
                self.to(device)
                self.eval()
                with torch.inference_mode():
                        y_pred = torch.argmax(nn.Softmax(self(X.to(device)), dim=1), dim=1)
                return y_pred


