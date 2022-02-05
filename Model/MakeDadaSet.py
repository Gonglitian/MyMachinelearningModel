import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from PIL import Image
from torch.utils.data import DataLoader

class makedataset(Dataset):
    def __init__(self,csv_filename,resize,mode):
        super(makedataset,self).__init__()
        self.csv_filename = csv_filename
        self.resize = resize
        self.image,self.label = self.load_csv()

        if mode == 'train':
            self.image = self.image[:int(0.6*len(self.image))]
            self.label = self.label[:int(0.6*len(self.label))]
        elif mode =='val':
            self.image = self.image[int(0.6*len(self.image)):int(0.8*len(self.image))]
            self.label = self.label[int(0.6*len(self.label)):int(0.8*len(self.label))]
        else:
            self.image = self.image[int(0.8*len(self.image)):]
            self.label = self.label[int(0.8*len(self.label)):]
    def load_csv(self):
        image,label = [],[]
        with open(self.csv_filename) as f:
            reader = csv.reader(f)
            for row in reader:
                i,l = row
                image.append(i)
                label.append(int(l))
        return image, label
    def __len__(self):
        return len(self.image)

    def __getitem__(self,idx):
        tf = transforms.Compose([lambda x : Image.open(x).convert('RGB'),
                                transforms.Resize(self.resize),
                                transforms.ToTensor()])
        image_tensor = tf(self.image[idx])
        label_tensor = torch.tensor(self.label[idx])
        return image_tensor, label_tensor

train_db = makedataset('myself_data.csv',(128,128),'train')

train_loder = DataLoader(train_db,batch_size=1)
for x,y in train_loder:
    print(x.shape)
    print(y.shape)