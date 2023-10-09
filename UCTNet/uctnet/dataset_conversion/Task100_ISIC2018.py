import json
from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/home/gxy/dataset/Task100_ISIC/')
    args = parser.parse_args()

    task_id = 100
    task_name = "ISIC2018"
    prefix = 'ISIC'

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    if isdir(imagestr):
        shutil.rmtree(imagestr)
        shutil.rmtree(imagests)
        shutil.rmtree(labelstr)
        shutil.rmtree(labelsts)

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    test_id = []
    train_id=[]
    f = open("/home/gxy/dataset/Task100_ISIC/dataset.json", 'r')
    content = f.read()
    split = json.loads(content)
    for ts in split['test']:
        ts_id = ts[16:23]
        test_id.append(ts_id)
        train_id.append(ts_id)
    for tr in split['training']:
        tr_id = tr['image'][16:23]
        train_id.append(tr_id)
    
    img_folder = join(args.dataset_path, "imagesTr")
    label_folder = join(args.dataset_path, "labelsTr")
    train_patient_names = []
    test_patient_names = []
    all_patients = subfiles(img_folder, join=False, suffix='0000.nii.gz')

    for p in all_patients:
        if p[5:12] in test_id:
            test_patient_name = f'{prefix}_{p[5:12]}.nii.gz'
            shutil.copy(join(img_folder, p), join(imagests,p))
            shutil.copy(join(img_folder, p[:-8]+"1.nii.gz"), join(imagests,p[:-8]+"1.nii.gz"))
            shutil.copy(join(img_folder, p[:-8]+"2.nii.gz"), join(imagests,p[:-8]+"2.nii.gz"))
            shutil.copy(join(label_folder,test_patient_name), join(labelsts, test_patient_name))
            test_patient_names.append(test_patient_name)
            
        if p[5:12] in train_id:
            train_patient_name = f'{prefix}_{p[5:12]}.nii.gz'
            shutil.copy(join(img_folder, p), join(imagestr,p))
            shutil.copy(join(img_folder, p[:-8]+"1.nii.gz"), join(imagestr,p[:-8]+"1.nii.gz"))
            shutil.copy(join(img_folder, p[:-8]+"2.nii.gz"), join(imagestr,p[:-8]+"2.nii.gz"))
            shutil.copy(join(label_folder,train_patient_name), join(labelstr, train_patient_name))
            train_patient_names.append(train_patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "ISIC"
    json_dict['description'] = "Skin legions segmentation"
    json_dict['tensorImageSize'] = "2D"
    json_dict['reference'] = "see ISIC 2018"
    json_dict['licence'] = "see ISIC 2018"
    json_dict['release'] = "1.0 04/05/2018"
    json_dict['modality'] = OrderedDict({
        "0": "R",
        "1": "G",
        "2": "B"}
    )
    json_dict['labels'] = OrderedDict({
        "0": "background",
        "1": "lesion"}
    )
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)

    json_dict['test'] = ["./imagesTs/%s" %
                         test_patient_name for test_patient_name in test_patient_names]

    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name,
                              "label": "./labelsTr/%s" % train_patient_name} for train_patient_name in train_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    
    ts_list = sorted(test_id,key=lambda keys:[ord(i) for i in keys],reverse=False)
    tr_list = sorted(train_id,key=lambda keys:[ord(i) for i in keys],reverse=False)
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = [f'{prefix}_{i}' for i in tr_list if i not in ts_list]
    splits[-1]['val'] = [f'{prefix}_{i}' for i in ts_list]
                         
    save_pickle(splits, join(out_base, "splits_final.pkl"))

if __name__ == "__main__":
   main()

    