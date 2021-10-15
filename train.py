import torch
import torch.utils.data
from torch import optim, nn
import torch.nn.functional as F

batch_size = 200
learning_rate = 0.001
epochs = 20

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 3),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x


net = MLP()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()
            ))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set : Averge loss: {:.4f}, Accurancy: {}/{}({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)
        ))