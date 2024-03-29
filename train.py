from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import visdom, torch, torchvision
from cartoongan import Generator, discriminator
from dataloader import image_dataset,ani_dataset
from torch.optim import Adam
import argparse
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
print(torch.cuda.get_device_name(0))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_normal(m.weight.data)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=24, help='input batch size')
    parser.add_argument('--imgdir', type=str, default="/home/yachao-li/Downloads/", help='path of  images')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lamda', type=float, default=0.1, help='the weight of perceptual loss')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPU')

    opt = parser.parse_args()
    print(opt)

    vis = visdom.Visdom()
    lineg = vis.line(Y=np.arange(10), env="G_Loss")
    lined = vis.line(Y=np.arange(10), env="D_Loss")

    real_image_dataset = image_dataset(path=opt.imgdir)
    ani_image_dataset = ani_dataset(path=opt.imgdir, type="ani_videos_moriyama")
    real_image_loader = DataLoader(real_image_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=2)
    ani_image_loader = DataLoader(ani_image_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=2, drop_last=True)

    G_net = Generator(in_dim=3).cuda()
    D_net = discriminator(in_dim=3).cuda()
    G_net.load_state_dict(torch.load("weights_in/G_init.pkl"))
    # G_net.apply(weights_init)
    # D_net.apply(weights_init)



    Vgg_model = torchvision.models.vgg19(pretrained=True)
    Vgg = nn.Sequential(*list(Vgg_model.features)[:26])
    Vgg.cuda()
    for param in Vgg.parameters():
        param.requires_grad = False

    if opt.ngpu > 1:
        G_net = nn.DataParallel(G_net)
        D_net = nn.DataParallel(D_net)
        Vgg = nn.DataParallel(Vgg)
    criterion = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()

    G_optimizer = Adam(G_net.parameters(), lr=opt.lr, betas=(0.5,0.99))
    D_optimizer = Adam(D_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
    G_net.train()
    D_net.train()
    G_loss_list = []
    D_loss_list = []
    g_loss_list = []
    per_loss_list = []
    d_loss_fakeani_list = []
    d_loss_realani_list = []
    for epoch in range(opt.epochs):
        for i, (real_img, ani_img) in enumerate(zip(real_image_loader, ani_image_loader)):

            real_img = Variable(real_img).cuda()

            real_ani = Variable(ani_img[0]).cuda()
            ani_edge = Variable(ani_img[1]).cuda()

            #train discriminator
            D_optimizer.zero_grad()

            real_aniout = D_net(real_ani)
            real_anilabel = Variable(torch.ones_like(real_aniout)).cuda()

            fake_anilabel = Variable(torch.zeros_like(real_aniout)).cuda()

            d_loss_realani = criterionMSE(real_aniout, real_anilabel)

            fake_ani = G_net(real_img)
            fake_aniout = D_net(fake_ani)
            d_loss_fakeani = criterionMSE(fake_aniout, fake_anilabel)

            edge_ani = D_net(ani_edge)
            d_loss_edge = criterionMSE(edge_ani, fake_anilabel)


            d_loss = d_loss_fakeani + d_loss_realani + d_loss_edge

            d_loss.backward()
            D_optimizer.step()


            #train generator
            G_optimizer.zero_grad()

            fake_ani = G_net(real_img)
            fake_aniout = D_net(fake_ani)
            g_loss = criterionMSE(fake_aniout, real_anilabel)
            #perceptual loss
            per_loss = criterion(Vgg(fake_ani),Vgg(real_img))

            G_loss =  g_loss + opt.lamda * per_loss

            G_loss.backward()
            G_optimizer.step()

            G_loss_list.append(G_loss.item())
            g_loss_list.append(g_loss.item())
            per_loss_list.append(per_loss.item())
            D_loss_list.append(d_loss.item())
            d_loss_fakeani_list.append(d_loss_fakeani.item())
            d_loss_realani_list.append(d_loss_realani.item())

            if i % 40 == 0:
                torchvision.utils.save_image((fake_ani), 'samples_mrym/' +"genepoch" + str(epoch+1) + "batch" + str(i + 1) + '.jpg', normalize=True)
                torchvision.utils.save_image((real_img), 'samples_mrym/' +"realepoch" + str(epoch+1) + "batch" + str(i + 1) + '.jpg', normalize=True)
                torchvision.utils.save_image((real_ani), 'samples_mrym/' +"aniepoch" + str(epoch+1) + "batch" + str(i + 1) + '.jpg', normalize=True)
                torchvision.utils.save_image((ani_edge), 'samples_mrym/' +"edgeepoch" + str(epoch+1) + "batch" + str(i + 1) + '.jpg', normalize=True)

                G_loss = sum(G_loss_list) / len(G_loss_list)
                D_loss = sum(D_loss_list) / len(D_loss_list)
                g_loss = sum(g_loss_list) / len(g_loss_list)
                per_loss = sum(per_loss_list) / len(per_loss_list)
                d_loss_realani = sum(d_loss_realani_list) / len(d_loss_realani_list)
                d_loss_fakeani = sum(d_loss_fakeani_list) / len(d_loss_fakeani_list)

                print("Epoch:{:.0f},Batch/Batchs: {:.0f}/{:.0f}, D_loss: {:.3f} d_loss_realani: {:.3f}  d_loss_fakeani: {:.3f} d_loss_edge: {:.3f}"
                      
                      "G_loss: {:.3f} g_loss: {:.3f} per_loss: {:.3f}".format(
                        epoch+1,i+1,len(ani_image_loader),
                        D_loss, d_loss_realani, d_loss_fakeani, d_loss_edge,
                        G_loss, g_loss,per_loss))
                vis.line(Y=np.column_stack((np.array(G_loss_list), np.array(g_loss_list), np.array(per_loss_list))),
                          X=np.column_stack((np.arange(len(G_loss_list)), np.arange(len(G_loss_list)), np.arange(len(G_loss_list) ))),
                          update="new",
                          opts=dict(title="G_loss",legend=["G_loss", "g_loss", "per_loss"]),  win=lineg,
                          env="G_Loss")
                #
                vis.line(Y=np.column_stack((np.array(D_loss_list),np.array(d_loss_fakeani_list), np.array(d_loss_realani_list))),
                          X=np.column_stack((np.arange(len(G_loss_list)), np.arange(len(G_loss_list)),
                                            np.arange(len(G_loss_list)))),
                          update="new",
                          opts=dict(title="D_loss",legend=["D_loss", "d_fake", "d_real"]),  win=lined,
                env="D_Loss")
        if (epoch+1) % 5 == 0:
            torch.save(G_net.state_dict(), "weights_mrym/" + str(epoch+1) + ".pkl")



if __name__ == '__main__':
    main()













