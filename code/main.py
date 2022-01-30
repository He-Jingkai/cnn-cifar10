import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional
import torch.optim
import ssl
from torch.utils.tensorboard import SummaryWriter
from model import Net

n_epochs = 100
ssl._create_default_https_context = ssl._create_unverified_context
writer = SummaryWriter('runs-ac=0.005-3layers-dropout-bn')
train_on_gpu = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
num_workers = 0
batch_size = 16
valid_size = 0.2
train_data = datasets.CIFAR10(
    'data', train=True,
    download=True, transform=transform
)
test_data = datasets.CIFAR10(
    'data', train=True, download=True, transform=transform
)
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
criterion = nn.CrossEntropyLoss()


def testAccuracy(loader, model):
    testLoss = 0.0
    classCorrect = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        testLoss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        for i in range(batch_size):
            label = target.data[i]
            classCorrect[label] += correct[i].item()
            class_total[label] += 1

    test_loss = testLoss / len(loader.dataset)
    writer.add_scalar('Loss of Test Data',
                      test_loss,
                      epoch)

    for i in range(10):
        className = classes[i]
        writer.add_scalar('Accuracy of ' + className + '/%',
                          100. * classCorrect[i] / class_total[i],
                          epoch)
    writer.add_scalar('Overall Accuracy/%',
                      100. * np.sum(classCorrect) / np.sum(class_total),
                      epoch)


model = Net()
print(model)
images = torch.randn(1, 3, 32, 32)
writer.add_graph(model, input_to_model=images, verbose=False)

if train_on_gpu:
    model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
train_loss_min = np.Inf

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    model.eval()
    writer.add_scalar('Loss of Train Data',
                      train_loss,
                      epoch)
    if (epoch - 1) % 5 == 0:
        testAccuracy(test_loader, model)
