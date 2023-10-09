import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import metrics

def dice_HD_score(output_folder, target_folder, class_num):
    dices, hds, ses, sps, ious, accs = 0,0,0,0,0,0
    mdices, mhds = np.zeros(1), np.zeros(1)
    mses, msps, mious, maccs = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    
    files = subfiles(output_folder, suffix=".nii.gz", join=False, sort=True)
    # print(files)
    print(len(files))
    patientDice, patientHD = np.zeros(len(files)), np.zeros(len(files))
    for i in range(len(files)):
        patient_seg_nii = sitk.ReadImage(join(output_folder,files[i]))
        patient_gt_nii = sitk.ReadImage(join(target_folder,files[i]))
        
        patient_seg_3D = sitk.GetArrayFromImage(patient_seg_nii)
        patient_gt_3D = sitk.GetArrayFromImage(patient_gt_nii)
        d, h, w = patient_seg_3D.shape
        # print("d, h, w", d, h, w)
        d1,h1,w1 = patient_gt_3D.shape
        # print("d1, h1, w1", d1, h1, w1)
        
        eval_label = 0
        for label in range(1, class_num):
            pred_i = np.zeros((d, h, w))
            pred_i[patient_seg_3D == label] = 1
            gt_i = np.zeros((d, h, w))
            gt_i[patient_gt_3D == label] = 1
            dice_i, hd95_i = metrics.calculate_metric_percase(pred_i, gt_i)
            SE_i, SP_i, IOU_i, Acc_i, _, _, _ = metrics.sespiou_coefficient(pred_i, gt_i)
            mdices[eval_label] += dice_i
            mhds[eval_label] += hd95_i
            mses[eval_label] += SE_i
            msps[eval_label] += SP_i
            mious[eval_label] += IOU_i
            maccs[eval_label] += Acc_i
            
            patientDice[i] += dice_i
            patientHD[i] += hd95_i
            eval_label += 1
            del pred_i, gt_i
    
    mdices = mdices / len(files)
    mhds = mhds / len(files)
    mses = mses / len(files)
    msps = msps / len(files)
    mious = mious / len(files)
    maccs = maccs / len(files)
    
    for i in range(1):
        dices += mdices[i] 
        hds += mhds[i]
        ses += mses[i] 
        sps += msps[i]
        ious += mious[i] 
        accs += maccs[i]
        
    print('mdices:',mdices)
    print('dices:',dices)
    print('mhds:',mhds)
    print('hds',hds)
    print('ses:',ses)
    print('sps:',sps)
    print('ious:',ious)
    print('accs:',accs)
    

if __name__ == "__main__":
    target_folder = "/home/gxy/dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task100_ISIC2018/labelsTs/"
    base_path = "/home/gxy/code/UCTNet-main/UCTNet/nnUNet_trained_models/nnUNet/2d/Task100_ISIC2018/UCTNetTrainer__nnUNetPlansv2.1/fold_0/"
    
    output_folder1 = base_path + "UCTNet" + "/model_final_checkpoint/"
    
    class_num = 2
    
    dice_HD_score(output_folder1, target_folder, class_num)


