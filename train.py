# python imports
import os
import glob
import warnings
# external imports
import pandas as pd
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from loss import ncc_loss
from loss import mse_loss
from loss import gradient_loss
from config import args
from data import Dataset
from NET import Affine, TwoStage
import torch.utils.data as data
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random


def setup_seed(seed):
    """ 该方法用于固定随机数

    Args:
        seed: 随机种子数

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):   # 求模型参数个数
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())  # 保存图片
    img.SetOrigin(ref_img.GetOrigin())  # 设置坐标原点
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())  # 设置间隔
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

def train():

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志
    log_name = 'AEAU_' + str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 3D 读入fixed图像
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    # 创建仿射网络（Affine）、配准网络（UNet）和STN
    TSregnet = TwoStage(2, 16, vol_size).to(device)
    TSregnet.train()
    # net.train()
    print('TSregnet:', TSregnet)
    # 模型参数个数
    print("TSregnet: ", count_parameters(TSregnet))

    # Set optimizer and loss
    opt = Adam(TSregnet.parameters(), lr=args.lr)

    sim_loss_fn = ncc_loss if args.sim_loss == "ncc" else mse_loss
    grad_loss_fn = gradient_loss

    # Get all the names of the training data
    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    Loss_list = []
    # Training loop.
    for i in range(1, args.n_iter + 1):
        # Generate the moving images and convert them to tensors. 生成运动图像，并且转换为向量
        input_moving = iter(DL).next()  # 迭代取图片

        # [B, C, D, W, H]
        input_moving = input_moving.to(device).float()

        # Run the data through the model to produce warp and flow field

        img = TSregnet(input_moving, input_fixed)
        flow = img['flow']
        affine_flow = img['affine_flow']
        affine_mov = img['affine_mov']
        warp = img['warp']
        # Calculate loss
        sim_loss = sim_loss_fn(warp, input_fixed)  # 相似性损失
        grad_loss_a = grad_loss_fn(flow)  # 平滑度损失
        # affine loss
        sim_loss_af = sim_loss_fn(affine_mov, input_fixed)  # 相似性损失
        loss = sim_loss + args.alpha * grad_loss_a + sim_loss_af
        print("i: %d  loss: %f  sim: %f  sim_aff: %f grad: %f" % (i, loss.item(), sim_loss.item(), sim_loss_af.item(), grad_loss_a.item()), flush=True)
        print("%d  %f  %f  %f" % (i, loss.item(), sim_loss.item(), grad_loss_a.item()), file=f)
        Loss_list.append(loss)
        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(TSregnet.state_dict(), save_file_name)
            m_name = str(i) + "_m.nii.gz"
            m2f_name = str(i) + "_m2f.nii.gz"
            save_image(input_moving, f_img, m_name)
            save_image(warp, f_img, m2f_name)
            print("warped images have saved.")
    f.close()
    Loss = pd.DataFrame([Loss_list])
    print(Loss)
    Loss.to_csv('./ACC/7/1tsnet.csv', index=False)
if __name__ == "__main__":

    setup_seed(12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()