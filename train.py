from torchvision import datasets, transforms, models
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import argparse
import my_functions as mf

# Set variables for executing the script
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, default='/home/workspace/paind-project/flowers')

parser.add_argument('--save_dir', action='store', 
                    dest='save_dir', default='/home/workspace/paind-project/checkpoint.pth')

parser.add_argument('--arch', action='store', 
                    dest='arch', default='alexnet')

parser.add_argument('--learning_rate', action='store', 
                    dest='lr', default='0.001')

parser.add_argument('--epochs', action='store', 
                    dest='epochs', default='10')

parser.add_argument('--hidden_units', action='store', 
                    dest='hu', default='1000')

parser.add_argument('--gpu', action='store_const',
                    dest='device',
                    const='cuda',
                   default='cpu')

args = parser.parse_args()

# Load the data
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# Define transforms so images are diverse
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(90),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load the datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Define dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False)

# Chose pre-trained model
model = models.__dict__[args.arch](pretrained=True)

# Freeze parameters so backpropagation process won't change them
for param in model.parameters():
    param.requires_grad = False
    
# Building classifier
if args.arch == 'alexnet':
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, 4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(4096, int(args.hu))),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(int(args.hu), 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
elif args.arch == 'vgg13':
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, int(args.hu))),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(int(args.hu), 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Underlaying the classifier for model
model.classifier = classifier

# Defining criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), float(args.lr))

# Defining training variables
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training the network on GPU!

model = mf.train_model(model, criterion, optimizer, scheduler, trainloader, validloader, train_data, valid_data, int(args.epochs), args.device)
            
# Save the model 
model.class_to_idx = train_data.class_to_idx
torch.save({'arch': args.arch,
           'state_dict': model.state_dict(),
           'class_to_idx': model.class_to_idx},
          args.save_dir)