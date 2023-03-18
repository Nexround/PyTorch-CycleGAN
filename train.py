#!/usr/bin/python3

import itertools
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from options.base_options import BaseOptions

from model.anime_gan import Generator
from model.discriminator import Discriminator
from model.vgg import VGG19
from utils.common import *
from datasets import ImageDataset

import wandb

if __name__ == '__main__':

    opt = BaseOptions().parse()

    if opt.use_wandb:
        # wandb setup
        wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt)
        columns = ["Epoch", "Real A", "Real B", "Fake A", "Fake B",
                   "Recovered A", "Recovered B", "Same A", "Same B"]
        result_table = wandb.Table(columns)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator()
    netG_B2A = Generator()
    netD_A = Discriminator(opt)
    netD_B = Discriminator(opt)

    nets = [netG_A2B, netG_B2A, netD_A, netD_B]

    torch.backends.cudnn.benchmark = True

    if opt.vgg:
        VGG = VGG19(init_weights=opt.vgg_model, feature_mode=True)
        VGG.eval()
        nets.append(VGG)

    for net in nets:
        net.to(torch.float16)
        if opt.cuda:
            net.cuda()
        if opt.mutil_gpu:
            net = torch.nn.DataParallel(net)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_content = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.HalfTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)
    # target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
    # 必须根据每个batch的大小来定义target_real的大小 否则可能会在最后一个batch出错，即数量不满足batch_size
    target_real = Tensor(opt.batch_size, 1).fill_(1.0)
    # target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)
    target_fake = Tensor(opt.batch_size, 1).fill_(0.0)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [transforms.RandomHorizontalFlip(),
                   transforms.Pad(30, padding_mode='edge'),
                   transforms.RandomRotation((-10, 10), ),
                   transforms.CenterCrop(opt.size),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   lambda x: x.to(torch.float16)]
    dataloader = DataLoader(ImageDataset(opt, transforms_=transforms_, unaligned=True),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    '''
    sample_param = next(netG_A2B.parameters())
    # 获取参数的数据类型
    param_dtype = sample_param.dtype
    print(f"模型参数的数据类型：{param_dtype}")
    '''
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for batch in tqdm(dataloader, desc='Epoch %d/%d' % (epoch, opt.n_epochs)):
            if batch['A'].shape[0] != opt.batch_size:
                """必须根据每个batch的大小来定义target_real的大小 否则可能会在最后一个batch出错，即数量不满足batch_size"""
                continue
            # Set model input

            real_A = batch['A'].to('cuda')
            real_B = batch['B'].to('cuda')

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Content loss
            if opt.vgg:
                x_feature = VGG((real_A + 1) / 2)
                G_feature = VGG((fake_B + 1) / 2)
                Con_loss = 10 * criterion_content(G_feature, x_feature.detach())
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + \
                loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + Con_loss

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + \
                loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

        if opt.use_wandb:
            img_dict = {"Real A": real_A, "Real B": real_B, "Fake A": fake_A, "Fake B": fake_B,
                        "Recovered A": recovered_A, "Recovered B": recovered_B, "Same A": same_A, "Same B": same_B}
            ims_dict = {}
            for label, image in img_dict.items():  # 遍历字典 使用.items()方法
                image_numpy = tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                ims_dict[label] = wandb_image
            img_list = ims_dict.values()
            result_table.add_data(epoch, *img_list)
            wandb.log({"fake_A": wandb.Image(fake_A),
                      "fake_B": wandb.Image(fake_B)})
            wandb.log({'Content_loss': Con_loss, 'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                       'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B), 'loss_GAN_A2B': loss_GAN_A2B, 'loss_GAN_B2A': loss_GAN_B2A, },
                      )

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'output/'+opt.name + '_netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/'+opt.name + '_netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/'+opt.name + '_netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/'+opt.name + '_netD_B.pth')
    ###################################
    if opt.use_wandb:
        wandb.save('output/netG_A2B.pth')
        wandb.save('output/netG_B2A.pth')
        wandb.log({"Log": result_table})
