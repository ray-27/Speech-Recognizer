import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming the model and create_masks are defined as provided in the previous example


# Setup data loaders
train_dataset = SpeechDataset()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpeechTransformer(num_features=128, num_classes=29).to(device)
criterion = nn.CTCLoss(blank=28).to(device)  # assuming 28 is the index of the blank label
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        input_lengths, target_lengths = torch.full((data.size(0),), data.size(1), dtype=torch.long), torch.full((target.size(0),), target.size(1), dtype=torch.long)
        output = model(data)
        output = output.log_softmax(2).permute(1, 0, 2)
        loss = criterion(output, target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
