import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import metrics

def dice_HD_score(output_folder, target_folder, class_num):
    dices = 0
    hds = 0
    mdices, mhds = np.zeros(8), np.zeros(8)
    
    files = subfiles(output_folder, suffix=".nii.gz", join=False, sort=True)
    print(files)
    print(len(files))
    patientDice, patientHD = np.zeros(len(files)), np.zeros(len(files))
    for i in range(len(files)):
        patient_seg_nii = sitk.ReadImage(join(output_folder,files[i]))
        patient_gt_nii = sitk.ReadImage(join(target_folder,files[i]))
        
        patient_seg_3D = sitk.GetArrayFromImage(patient_seg_nii)
        patient_gt_3D = sitk.GetArrayFromImage(patient_gt_nii)
        d, h, w = patient_seg_3D.shape
        d1,h1,w1 = patient_gt_3D.shape
        
        eval_label = 0
        for label in range(1, class_num):
            if label in [10, 12, 13, 5, 9]:
                continue
            pred_i = np.zeros((d, h, w))
            pred_i[patient_seg_3D == label] = 1
            gt_i = np.zeros((d, h, w))
            gt_i[patient_gt_3D == label] = 1
            dice_i, hd95_i = metrics.calculate_metric_percase(pred_i, gt_i)
            mdices[eval_label] += dice_i
            mhds[eval_label] += hd95_i
            patientDice[i] += dice_i
            patientHD[i] += hd95_i
            eval_label += 1
            del pred_i, gt_i
    
    mdices = mdices / len(files)
    mhds = mhds / len(files)
    for i in range(8):
        dices += mdices[i] 
        hds += mhds[i]
    
    mdices_  = [mdices[6],mdices[3],mdices[2],mdices[1],mdices[4],mdices[7],mdices[0],mdices[5]]
    mhds_ = [mhds[6],mhds[3],mhds[2],mhds[1],mhds[4],mhds[7],mhds[0],mhds[5]]
    
    print('patientDice',patientDice/8)
    print('patientHD',patientHD/8)
    print('mdices:',mdices_)
    print('dices:',dices / (8))
    print('mhds:',mhds_)
    print('hds',hds / (8))
    

if __name__ == "__main__":
    target_folder = "/home/gxy/dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task017_AbdominalOrganSegmentation/labelsTs/"
    base_path = "/home/gxy/code/UCTNet-main/UCTNet/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/UCTNetTrainer__nnUNetPlansv2.1/fold_0/"
    
    output_folder1 = base_path + "UCTNet" + "/model_final_checkpoint/"
    class_num = 14
    
    dice_HD_score(output_folder1, target_folder, class_num)


