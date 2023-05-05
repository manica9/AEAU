import os
import glob
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import loss
from config import args
import matplotlib.pyplot as plt
from NET import Affine, TwoStage, SpatialTransformer
from medpy import metric
def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

def group_dice():
    FL = [i for i in range(21, 35, 1)]
    PL = [i for i in range(41, 51, 1)]
    OL = [i for i in range(61, 69, 1)]
    TL = [i for i in range(81, 93, 1)]
    CL = [101, 102, 121, 122]
    Ptm = [163, 164]
    Hpcp = [165, 166]
    group_ind = [FL, PL, OL, TL, CL, Ptm, Hpcp]
    return group_ind

def compute_label_group_dice(gt, pred, group):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    group_lst = []
    for ind in group:
        cls_lst = ind  # FL PL OL TL CL PTM HPCP
        dice_lst = []
        for cls in cls_lst:
            dice = loss.DSC(gt == cls, pred == cls)
            dice_lst.append(dice)
        group_lst.append(np.mean(dice_lst))  # 每个组的平均dice
    return group_lst

# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)

    f_img = sitk.ReadImage(args.atlas_file)  # 读取固定图像
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # set up atlas tensor
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))  # 匹配所有test下面.nii.gz的文件
    print("The number of test data: ", len(test_file_lst))

    # Set up model
    UNet = TwoStage(2, 16, vol_size).to(device)
    UNet.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    DSC = []
    JAC = []
    ASD = []
    HD = []
    J = []
    DET = []
    # fixed图像对应的label
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))  # S01.delineation.structure.label.nii.gz OASIS_OAS1_0002_MR1aligned_seg35.nii
    for file in test_file_lst:
        with torch.no_grad():
            name = os.path.split(file)[1]
            input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
            input_moving = torch.from_numpy(input_moving).to(device).float()
            # 读入moving图像对应的label
            label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0] # lpba
            input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
            input_label = torch.from_numpy(input_label).to(device).float()
            # 获得配准后的图像和label
            pred_flow = UNet(input_moving, input_fixed)
            pred_label = STN_label(input_label, pred_flow['flow'])
            pred_img = STN_img(input_moving, pred_flow['flow'])
            # 计算DSC
            dice = compute_label_group_dice(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy(), group_dice())
            print("dice: ", dice)
            DSC.append(dice)

            # jacobin
            jac_det = loss.Get_Ja(pred_flow['flow'].permute(0, 2, 3, 4, 1))
            c = (jac_det < 0).nonzero()
            per = (c.size(0) / (jac_det.size(1) * jac_det.size(2) * jac_det.size(3)) * 100)
            print("c: ", c.size(0))
            print("jac: ", per)
            JAC.append(per)
            J.append(c.size(0))
            DET.append(jac_det.cpu().detach().numpy())
        # if '7' in file:
        #     save_image(pred_img, f_img, "AEAU_7lp_warped.nii.gz")
        #     save_image(pred_flow['flow'].permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, "AEAU_7lp_flow.nii.gz")
        #     save_image(pred_label, f_img, "AEAU_7lp_label.nii.gz")
        # del pred_flow, pred_img, pred_label

    print(DSC)
    print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))
    print("mean: JAC", np.mean(JAC))
    print("mean: J", np.mean(J))
    print("std: DET", np.std(DET))
if __name__ == "__main__":
    test()
