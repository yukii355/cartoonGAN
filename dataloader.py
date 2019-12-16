import torch
from torch.utils.data import DataLoader
import cv2
import os
from torchvision import transforms
from PIL import Image
def image_list(path="~/Downaloads/", type="real/4"):
    data_list = []
    file_list = os.listdir(path + type)
    file_list.sort(key=lambda x: int(x[:-4]))

    if type == "real/4":
        for image in file_list:
            data_list.append(os.path.join(path + type +"/" , image ))
    else :
        for image in sorted(os.listdir(path + type)):
            data_list.append([os.path.join(path + type+"/" , image ),os.path.join(path + type+"_edge/", image)])

    return data_list




transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class image_dataset(DataLoader):

    def __init__(self, path, transforms=transform, type = "real/4") :

        self.real_image_list = image_list(path=path, type=type)
        print(self.real_image_list)
        self.transforms = transforms
        print("# of training real images samples:", len(self.real_image_list))

    def __getitem__(self, index):

        img_path_list = self.real_image_list[index]

        img = cv2.imread(img_path_list)[:,:,(2,1,0)]

        return  self.transforms(img), os.path.split(img_path_list)[1]

    def __len__(self):
        return len(self.real_image_list)

class ani_dataset(DataLoader):

    def __init__(self, path, transforms=transform, type = "ani_videos_moriyama") :

        self.ani_image_list = image_list(path=path, type=type)
        self.transforms = transforms
        print("# of training ani images samples:", len(self.ani_image_list))

    def __getitem__(self, index):

        img_path_list = self.ani_image_list[index]

        img = cv2.imread(img_path_list[0])[:,:,(2,1,0)]

        img_edge = cv2.imread(img_path_list[1])[:,:,(2,1,0)]


        return  self.transforms(img), self.transforms(img_edge)

    def __len__(self):
        return len(self.ani_image_list)


# data = image_dataset(path="/home/yachao-li/Downloads/")
# dataloader = torch.utils.data.DataLoader(data,128)
# for i in dataloader:
#     print(i.size())



