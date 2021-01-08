import numpy as np
import torch
import torch.nn as nn
import torch,torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torch import optim
from torchsummary import summary
from PIL import Image
from torch.utils.tensorboard import SummaryWriter # tensorboard
writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FoodDataset(Dataset):
    
    def __init__(self, filenames, labels, transform):
        
        self.filenames = filenames 
        self.labels = labels 
        self.transform = transform 
 
    def __len__(self):
        
        return len(self.filenames) 
 
    def __getitem__(self, idx):
        
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image) 
        label = np.array(self.labels[idx])
      
        return image,label 

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
 
test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def Split_Dataset(data_dir):
    
    dataset = ImageFolder(data_dir) 
    
    # create 101 labels
    character = [[] for i in range(len(dataset.classes))]
    
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): 
        np.random.seed(45)
        np.random.shuffle(data)
            
        # -------------------------------------------
        # seperate datasets into training & testing
        # -------------------------------------------
        
        num_sample_train = round(0.8*len(data))
        # training datasets  
        for x in data[0:num_sample_train] : 
            train_inputs.append(x)
            train_labels.append(i)
        # testing datasets
        for x in data[num_sample_train:] : 
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(FoodDataset(train_inputs, train_labels, train_transformer),
                                  batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(FoodDataset(test_inputs, test_labels, test_transformer),
                                  batch_size = batch_size, shuffle = False)
    
    return train_dataloader, test_dataloader

class BuildModel(nn.Module):

    def __init__(self):
        
        super(BuildModel, self).__init__()
        
        # ----------------------------------------------
        self.n = models.resnet50(pretrained = True).to(device)
        self.fc1 = nn.Linear(1000,101)
        # ----------------------------------------------
               
    def forward(self, x):
        
        # ----------------------------------------------
        x = self.n(x)
        x = self.fc1(x)
        # ----------------------------------------------
        return x

# parameters
batch_size = 64
lr = 1e-3
epochs = 30

# image path
data_dir = 'food\\images'
train_dataloader, test_dataloader = Split_Dataset(data_dir)

#bulid model 
C =  BuildModel().to(device)
optimizer_C = optim.SGD(C.parameters(), lr = lr)

# choose lose function
criteron = nn.CrossEntropyLoss()

loss_epoch_C = []
train_acc, test_acc = [], []

if __name__ == '__main__':    

    # print network architecture
    print(summary(C,(3,224,224)))
    
    for epoch in range(epochs):
    
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        # ----------------------------------------------------------------
        # Training Stage

        C.train() # 設定 train 或 eval
      
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))  
           
        for i, (x, label) in enumerate(train_dataloader) :

            
            x, label = x.to(device), label.to(device)
            optimizer_C.zero_grad() # init grad
            output = C(x) 
            
            loss = criteron(output, torch.as_tensor(label,dtype=torch.long)) 
            loss.backward() 
            optimizer_C.step() # update weight
            
            _, predicted = torch.max(output.data, 1)
            total_train += len(x)
            correct_train += (predicted == label).sum()
            train_loss_C += loss.item()
            iter += 1    

        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % \
              (epoch + 1, train_loss_C / iter, correct_train / total_train))
        
        writer.add_scalar("Loss/train", train_loss_C / iter,epoch) # write to tensorboard
        writer.add_scalar("Accuracy/train", correct_train / total_train,epoch) # write to tensorboard
        #----------------------------------------------------------------


        # ---------------------------------------------------------------
        # Testing Stage  
        C.eval() 
          
        for i, (x, label) in enumerate(test_dataloader) :
          
            with torch.no_grad(): 
                x, label = x.to(device), label.to(device)
                
                output=C(x) 
                _, predicted = torch.max(output.data, 1)
                total_test += len(x)
                correct_test += (predicted == label).sum()
        
        print('Testing acc: %.3f' % (correct_test / total_test))
                                     
        train_acc.append(100 * correct_train / total_train) 
        test_acc.append(100 * correct_test / total_test)  
        loss_epoch_C.append(train_loss_C/iter)
        #----------------------------------------------------------------
    torch.save(C,'resnet50_epoch60.pt')
    writer.close()