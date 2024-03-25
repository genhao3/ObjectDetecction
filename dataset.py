from torch.utils.data import Dataset,DataLoader
import pandas as pd
import cv2,os,torch
from torchvision import transforms


class VOCDataset(Dataset):
    def __init__(self,is_train=True):
        self.is_train = is_train

        self.traintxt = 'F:\\360Downloads\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main\\train.txt'
        self.val_txt = 'F:\\360Downloads\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Main\\val.txt'
        self.img_path = 'F:\\360Downloads\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
        self.label_path = 'F:\\360Downloads\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Detectionlabels'

        self.tmp = list(pd.read_csv(self.traintxt if is_train else self.val_txt, names=['filenames']).values.reshape(-1))
        self.filenames = []
        self.aug = transforms.Compose([
            transforms.ToTensor()
        ])


        # 去除空label
        i = 0
        for filename in self.tmp:
            f = os.path.getsize(os.path.join(self.label_path,filename+'.txt'))
            if f:
                
                self.filenames.append(filename)
            else:
                # print(filename,f)
                i += 1

        print('miss {} label'.format(i))



    

    def __len__(self):

        return len(self.filenames)
    
    def box_center_to_corner(self,boxes):
        """从（中间，宽度，高度）转换到（左上，右下）"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes = torch.stack((x1, y1, x2, y2), axis=-1)

        return boxes
    

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        for b in batch:
            if b[1].dtype == torch.float32:
                images.append(b[0])
                boxes.append(b[1])
        images = torch.stack(images, dim=0)
        return images, boxes


    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.img_path,self.filenames[index])+'.jpg')

        h, w = image.shape[0:2]
        image = cv2.resize(image, (224, 224))
        
        
        image = self.aug(image)

        bbox = pd.read_csv(os.path.join(self.label_path, self.filenames[index] + '.txt'), names=['labels', 'x', 'y', 'w', 'h'],sep=' ').values
        # if bbox.dtype != 'float64':
        #     print(self.filenames[index],bbox.dtype)
        # if bbox.dtype == 'float64':
        bbox = torch.tensor(bbox, dtype=torch.float32)
        label = bbox[:, 0].reshape(-1, 1)
        bbox = self.box_center_to_corner(bbox[:, 1:]) # 把中心转为四个角
        bbox = torch.cat((label, bbox), dim=1)

        return image, bbox
    


if __name__ == '__main__':
    train = VOCDataset(is_train=True)
    val = VOCDataset(is_train=False)
    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2,pin_memory=True,collate_fn=train.collate_fn)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=2,pin_memory=True,collate_fn=val.collate_fn)

    for img,label in train_loader:
        print(label)

