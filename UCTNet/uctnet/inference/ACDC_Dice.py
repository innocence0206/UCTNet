import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import metrics

def dice_score(output_folder, target_folder, class_num):
    dices = 0
    mdices, mhds = np.zeros(4), np.zeros(4)

    files = subfiles(output_folder, suffix=".nii.gz", join=False, sort=True)
    maybe_patient_ids = np.unique([i[:11] for i in files])
    print(len(maybe_patient_ids))
    patientDice = np.zeros(len(maybe_patient_ids))

    for i in range(len(maybe_patient_ids)):
        patient_i_seg1_nii = sitk.ReadImage(join(output_folder,files[2*i]))
        patient_i_seg2_nii = sitk.ReadImage(join(output_folder,files[2*i+1]))

        patient_i_gt1_nii = sitk.ReadImage(join(target_folder,files[2*i]))
        patient_i_gt2_nii = sitk.ReadImage(join(target_folder,files[2*i+1]))
        
        patient_i_seg1_3D = sitk.GetArrayFromImage(patient_i_seg1_nii)
        patient_i_seg2_3D = sitk.GetArrayFromImage(patient_i_seg2_nii)
        patient_i_gt1_3D = sitk.GetArrayFromImage(patient_i_gt1_nii)
        patient_i_gt2_3D = sitk.GetArrayFromImage(patient_i_gt2_nii)

        seg_patient = np.concatenate((patient_i_seg1_3D[None, :, :, :], patient_i_seg2_3D[None, :, :, :]), axis=0)
        gt_patient = np.concatenate((patient_i_gt1_3D[None, :, :, :], patient_i_gt2_3D[None, :, :, :]), axis=0)

        for label in range(1, class_num):
            c, d, h, w = seg_patient.shape
            # print("c, d, h, w",c, d, h, w)
            pred_i = np.zeros((c, d, h, w))
            pred_i[seg_patient == label] = 1
            gt_i = np.zeros((c, d, h, w))
            gt_i[gt_patient == label] = 1
            dice_i, hd95_i = metrics.calculate_metric_percase(pred_i, gt_i)
            mdices[label] += dice_i
            patientDice[i] += dice_i
            
            del pred_i, gt_i
    print('patientDice',patientDice/3)
    mdices = mdices / len(maybe_patient_ids)
    for i in range(1, class_num):
        dices += mdices[i] 
    print('mdices:',mdices)
    print('dices:',dices / (class_num - 1))


  
    

if __name__ == "__main__":
    base_path_2D = "/home/gxy/code/UCTNet-main/UCTNet/nnUNet_trained_models/nnUNet/2d/Task027_ACDC/UCTNetTrainer__nnUNetPlansv2.1/fold_0/" 
    target_folder = "/home/gxy/dataset/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC/labelsTs/"
    
    output_folder1 = base_path_2D + "UCTNet" + "/model_final_checkpoint/"
    
    class_num = 4
    dice_score(output_folder1, target_folder, class_num)


