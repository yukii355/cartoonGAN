
from torch.autograd import Variable
from torch.utils.data import DataLoader
import  torch, torchvision
from cartoongan import Generator
from dataloader import image_dataset

import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
print(torch.cuda.get_device_name(0))

def main():
    real_image_dataset = image_dataset(path="/home/yachao-li/Downloads/")
    real_image_loader = DataLoader(real_image_dataset, batch_size=1,  num_workers=0)
    G_net = Generator(in_dim=3).cuda()
    G_net.load_state_dict(torch.load("weights_edge/30.pkl"))

    for i, real_img in enumerate(real_image_loader):
        real_img_ = Variable(real_img[0]).cuda()
        print(real_img[1][0])

        fake_ani = G_net(real_img_)
        if not os.path.exists("test_4"):
            os.mkdir("test_4")
        torchvision.utils.save_image((fake_ani),
                                     'test_4/' + real_img[1][0],
                                     normalize=True)



if __name__ == '__main__':
    main()