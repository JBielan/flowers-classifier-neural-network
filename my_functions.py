from torchvision import models
import torch
from collections import OrderedDict
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import copy

def train_model(model, criteria, optimizer, scheduler, trainloader, validloader, train_data, valid_data, epochs, device):
    model.to(device)
    best_acc = 0
    for epoch in range(1, epochs+1):
        print('*'*15)
        print('Epoch {} of {}'.format(epoch, epochs))
        print('*'*15)
        
        # Dividing whole process to training and validation data
        for data in [trainloader, validloader]:
            if data == trainloader:
                scheduler.step()
                model.train()   # Training mode
            else:
                model.eval()    # Validation mode

            current_loss = 0
            current_corrects = 0

            # Load data
            for inputs, labels in data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(data == trainloader):

                    # Give input and prepare for training
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criteria(outputs, labels)

                    # Train if data from trainloader
                    if data == trainloader:
                        loss.backward()
                        optimizer.step()

                # Preparing stats for summary
                current_loss += loss.item()*inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            # Printing summary
            if data == trainloader:
                epoch_loss = current_loss / len(train_data)
                epoch_acc = current_corrects.double() / len(train_data)

                print('Train loss: {} , Train Accuracy: {}'.format(epoch_loss, epoch_acc))

            if data == validloader:
                epoch_loss = current_loss / len(valid_data)
                epoch_acc = current_corrects.double() / len(valid_data)

                print('Valid loss: {} , Valid Accuracy: {}'.format(epoch_loss, epoch_acc))
                
            # Choose the best model basing on validation 
            if data == validloader and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                
    # Load best model weights
    model.load_state_dict(best_state_dict)
    return model

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
    elif chpt['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
    else:
        print('Oops, base architecture should be alexnet or vgg13')
        
    model.class_to_idx = chpt['class_to_idx']
    
    if chpt['arch'] == 'alexnet':
        classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, 4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(1000, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    elif chpt['arch'] == 'vgg13':
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, int(1000))),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(int(1000), 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
    elif chpt['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
    else:
        print('Oops, base architecture should be alexnet or vgg13')
        
    model.class_to_idx = chpt['class_to_idx']
    
    if chpt['arch'] == 'alexnet':
        classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(9216, 4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(1000, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    elif chpt['arch'] == 'vgg13':
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, int(1000))),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(int(1000), 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model

def process_image(image):
  
    
    # Resizing with shorter side 
    new_h = 0
    new_w = 0
    im = image
    ratio = im.height/im.width
        
    if im.height < im.width:
        new_h = 256
        new_w = int(256/ratio)
    elif im.height > im.width:
        new_w = 256
        new_h = int(256*ratio)
    else:
         im.thumbnail((256, 256), Image.ANTIALIAS)
        
    size = new_w, new_h
    im = im.resize(size)
    
    # Croping PIL image
    width, height = im.size
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    im = im.crop((left, top, right, bottom))
    
    # Normalizing color channels encoding to 0-1 floats
    np_image = np.array(im)
    np_image = np_image/255
    
    # Substructing means and dividing by std dev (?)
    means = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    np_image = (np_image - means) / std_dev
    
    # Changing color dimension to index 0
    np_image = np_image.transpose(2,0,1)
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, label_map, topk):    
   
    # Processing the image
    img = Image.open(image_path)
    img = process_image(img)
    
    # Converting Numpy to Tensor
    img_t = torch.from_numpy(img).type(torch.FloatTensor)
    input_image = img_t.unsqueeze(0)
    
    # Probabilities   ---   the moment with some mistakes (?) RuntimeError: size mismatch, m1: [672 x 224], m2: [9216 x 4096]

    probs = torch.exp(model.forward(input_image))
    
    # Getting 5 top probabilities
    top_p, top_l = probs.topk(int(topk))
    top_p = top_p.detach().numpy().tolist()[0]
    top_l = top_l.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_flowers = [label_map[idx_to_class[lab]] for lab in top_l]
    
    # Returning top probabilities and top classes
    return top_p, top_flowers