import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Compute loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Compute loss
        acc = accuracy(out, labels)  # Compute accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Mean loss
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Mean accuracy
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            # Adjusted input size based on 150x150 input images (256 x 18 x 18)
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, xb):
        return self.network(xb)
    

class Trainer():
    def __init__(self):
        self.model = NaturalSceneClassification()
        self.history = []
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        outputs = [self.model.validation_step(batch) for batch in val_loader]
        return self.model.validation_epoch_end(outputs)
    
    def fit(self, epochs, lr, train_loader, val_loader, opt_func=torch.optim.SGD):
        self.history = []
        optimizer = opt_func(self.model.parameters(), lr)
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                loss = self.model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            self.model.epoch_end(epoch, result)
            self.history.append(result)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {result['train_loss']:.4f} | Val Loss: {result['val_loss']:.4f} | Val Acc: {result['val_acc']:.4f}\n")
        return self.history
    
    def plot_accuracy(self):
        val_acc = [x['val_acc'] for x in self.history]
        plt.plot(val_acc, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')
        plt.savefig('accuracy_plot.png')
        
    def plot_loss(self):
        train_loss = [x['train_loss'] for x in self.history]
        val_loss = [x['val_loss'] for x in self.history]
        plt.plot(train_loss, '-bx')
        plt.plot(val_loss, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss vs. No. of epochs')
        plt.legend(['Training', 'Validation'])
        plt.savefig('loss_plot.png')
            
    def get_model(self):
        return self.model
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)