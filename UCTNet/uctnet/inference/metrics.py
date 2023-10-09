import glob
import os
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from medpy import metric
from nnunet.paths import nnUNet_raw_data
import pandas as pd


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def get_dice_hd95(pre_path, experiment_id, task_id):
    task = convert_id_to_task_name(task_id)
    label_path = join(nnUNet_raw_data, task)
    predict_list = sorted(glob.glob(os.path.join(pre_path, '*nii.gz')))
    label_list = sorted(
        glob.glob(os.path.join(label_path, 'labelsTs', '*nii.gz')))

    print("loading success...")
    dataset = load_json(join(label_path, 'dataset.json'))
    metric_list = 0.0
    for predict, label in zip(predict_list, label_list):
        case = predict.split('/')[-1]
        print(case)
        predict, label, = read_nii(predict), read_nii(label)
        metric_i = []
        for i in dataset['evaluationClass']:
            metric_i.append(calculate_metric_percase(predict == i, label == i))
        metric_list += np.array(metric_i)
        print('case %s mean_dice %f mean_hd95 %f' %
              (case, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(predict_list)
    clm = ["DSC", "HD", "Aotra", "Gallbladder", "Kidnery(L)", "Kidnery(R)", "Liver", "Pancreas",
           "Spleen", "Stomach"] if task_id == 17 else ["DSC", "HD", "RV", "MLV", "LVC"]
    for i in range(len(dataset['evaluationClass'])):
        print('Mean class %s mean_dice %f mean_hd95 %f' %
              (clm[i+2], metric_list[i][0], metric_list[i][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (
        performance, mean_hd95))

    idx = experiment_id
    data = np.hstack((performance*100, mean_hd95,
                     metric_list[:, 0]*100)).reshape(1, len(dataset['evaluationClass'])+2)
    df = pd.DataFrame(data, index=[idx], columns=clm)
    df.to_csv(join(pre_path, f"{experiment_id}_result.cvs"))

    df.to_excel(
        join(pre_path, f"{experiment_id}_result.xlsx"), sheet_name='Synapse')

def sespiou_coefficient(pred, gt, smooth=1e-5):
    """ computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    #pred_flat = pred.view(N, -1)
    #gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth)/(TP + FP + FN + TN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2*Precision*Recall/(Recall + Precision +smooth)
    return SE.sum() / N, SP.sum() / N, IOU.sum() / N, Acc.sum()/N, F1.sum()/N, Precision.sum()/N, Recall.sum()/N

