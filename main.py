from img_processing import load_data, predict_from_file
from models import Trainer
from torch import optim
import torch

# print("Loading data...")
# train_loader, val_loader, test_loader = load_data()

n_epochs = 30
opt_func = optim.Adam
lr = 0.001

# print("Training model...")
trainer = Trainer()
# history = trainer.fit(n_epochs, lr, train_loader, val_loader, opt_func)
# trainer.plot_accuracy()
# trainer.plot_loss()
model = trainer.get_model()
# trainer.save_model("natural_scene_classification.pth")

model.load_state_dict(torch.load('./natural_scene_classification.pth', map_location=torch.device('cpu')))
model.eval()

predict_from_file('images/glacier_scenery.jpg', model)


