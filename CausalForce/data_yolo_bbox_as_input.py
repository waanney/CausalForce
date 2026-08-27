import os
import numpy as np
import torch 
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class MultipleRisksDataset(Dataset):
    
    def __init__(self, data_root):
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.front_imgs = []
        self.labels = []
        self.risk_ids = []
        self.objects_list = []
        self.target_img = []
        self.scenario_id_list = []
        self.risk_interval_H8_list = []
        self.yolo_bbox = []

        scenarios = sorted(
            name for name in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, name))
        )
        for scenario in scenarios:
            scenario_root = os.path.join(self.data_root, scenario)
            image_root = os.path.join(scenario_root, 'rgb_front')
            if not os.path.isdir(image_root):
                continue
            images = sorted(
                name for name in os.listdir(image_root)
                if os.path.isfile(os.path.join(image_root, name))
            )
            
            for i in range(0, len(images)-2):
                front_imgs = []
                objects = []
                for j in range(i, i+3):
                    frame_id = os.path.splitext(images[j])[0]
                    front_imgs.append(os.path.join(image_root, images[j]))
                    objects.append(os.path.join(scenario_root, 'detection_yolo', frame_id + '.npy'))
                self.front_imgs.append(front_imgs)
                self.objects_list.append(objects)
                
                target_frame_id = os.path.splitext(images[i+2])[0]
                self.labels.append(os.path.join(scenario_root, '2d_bbs_front', target_frame_id + '.npy'))
                self.risk_interval_H8_list.append(os.path.join(scenario_root, 'risk_interval_H8_new', target_frame_id + '.npy'))
                self.target_img.append(os.path.join(image_root, images[i+2]))
                self.risk_ids.append(os.path.join(scenario_root, 'risk_id.json'))
                self.scenario_id_list.append(scenario)


    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.labels)

    def __getitem__(self, index):

        data = dict()
        
        imgs = []
        all_objects = []
        all_objects_id = []

        for i in range(3):
            img = Image.open(self.front_imgs[index][i]).convert("RGB")
            if self._im_transform:
                img = self._im_transform(np.array(img))
            imgs.append(img)

            bbs_2d_front = np.load(self.objects_list[index][i], allow_pickle=True)
            objects, ids = self.get_all_objects(bbs_2d_front)

            if len(objects) == 0:
                objs_tensor = torch.empty((0, 4))

            else:
                objs_tensor = torch.tensor(objects, dtype=torch.float)

            all_objects.append(objs_tensor)
            all_objects_id.append(ids)

        input_tensor = torch.stack(imgs)

        with open(self.risk_ids[index]) as f:
            risk_ids_js = json.load(f)

        bbs_2d_front = np.load(self.labels[index], allow_pickle=True)
        

        _, risk_bbs = self.get_label_from_bbs(risk_ids_js, bbs_2d_front)

        if len(risk_bbs) == 0:
            risk_bbs_tensor = torch.empty((0, 4))
        else:
            risk_bbs_tensor = torch.tensor(risk_bbs, dtype=torch.float)

        
        risk_interval_H8 = np.load(self.risk_interval_H8_list[index], allow_pickle=True).item()
        risk_ids = [int(key) for key in risk_interval_H8.keys()]
        risk_type_by_id = self.get_risk_type_by_id(risk_ids_js)
        missing_types = [risk_id for risk_id in risk_ids if risk_id not in risk_type_by_id]
        if missing_types:
            raise ValueError(
                f"Risk IDs {missing_types} in {self.risk_interval_H8_list[index]} "
                f"are missing from {self.risk_ids[index]}"
            )
        risk_type_tensor = torch.tensor(
            [risk_type_by_id[risk_id] for risk_id in risk_ids], dtype=torch.long
        )
        risk_interval_H8_tensor = [
            torch.tensor(risk_interval_H8[str(item)], dtype=torch.float)
            for item in risk_ids
        ]

        data['target_img'] = Image.open(self.target_img[index]).convert("RGB")
        data['front_imgs'] = input_tensor
        data['label_risk_type'] = risk_type_tensor
        data['label_risk_bbs'] = risk_bbs_tensor
        data['all_objs_bbs'] = all_objects
        data['scenario_id'] = self.scenario_id_list[index]
        data['risk_interval_H8'] = risk_interval_H8_tensor
        data['risk_id'] = risk_ids
        data['all_objs_id'] = all_objects_id

        return data 

    @staticmethod
    def get_risk_type_by_id(risk_ids):
        category_to_index = {"Obs": 0, "Occ": 1, "I": 2, "C": 3}
        result = {}
        for category, raw_ids in risk_ids.get('risk_id', []):
            if category not in category_to_index:
                continue
            object_ids = raw_ids if isinstance(raw_ids, (list, tuple)) else [raw_ids]
            for object_id in object_ids:
                result[int(object_id)] = category_to_index[category]
        return result
    

    def get_all_objects(self, bbs_2d_front):
        
        all_objs = []
        all_ids = []
        
        for bb in bbs_2d_front.item()['boxes']:
            all_objs.append(bb)

        return all_objs, all_ids
    
    
    def get_label_from_bbs(self, risk_ids, bbs_2d_front):
        
        # 0: obstacle
        # 1: occlusion
        # 2: interaction
        # 3: collision
        
        risk_type = []
        risk_bbs = []
        

        for i in range(len(risk_ids['risk_id'])):

            # Obs
            if risk_ids['risk_id'][i][0] == "Obs":
                for id, bb in bbs_2d_front.item()['obstacles']:
                    if id in risk_ids['risk_id'][i][1]:
                        risk_type.append(0)
                        risk_bbs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])
                    
            # I
            if risk_ids['risk_id'][i][0] == "I":
                for id, bb in bbs_2d_front.item()['vehicles']:
                    if id == risk_ids['risk_id'][i][1]:
                        risk_type.append(2)
                        risk_bbs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])
                    
            # C
            if risk_ids['risk_id'][i][0] == "C":
                for id, bb in bbs_2d_front.item()['vehicles']:
                    if id == risk_ids['risk_id'][i][1]:
                        risk_type.append(3)
                        risk_bbs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])
                    
            # Occ
            if risk_ids['risk_id'][i][0] == "Occ":
                for id, bb in bbs_2d_front.item()['vehicles']:
                    if id == risk_ids['risk_id'][i][1][0]:
                        risk_type.append(1)
                        risk_bbs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])

                    if id == risk_ids['risk_id'][i][1][1]:
                        risk_type.append(1)
                        risk_bbs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])

                for id, bb in bbs_2d_front.item()['pedestrians']:
                    if id == risk_ids['risk_id'][i][1][1]:
                        risk_type.append(1)
                        risk_bbs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])

        return risk_type, risk_bbs

def custom_collate_fn(batch):
    collated = {}
    
    collated['front_imgs'] = torch.stack([item['front_imgs'] for item in batch], dim=0)
    collated['label_risk_type'] = [item['label_risk_type'] for item in batch]
    collated['label_risk_bbs'] = [item['label_risk_bbs'] for item in batch]
    collated['all_objs_bbs'] = [item['all_objs_bbs'] for item in batch]
    collated['all_objs_id'] = [item['all_objs_id'] for item in batch]
    collated['target_img'] = [item['target_img'] for item in batch]
    collated['scenario_id'] = [item['scenario_id'] for item in batch]
    collated['risk_interval_H8'] = [item['risk_interval_H8'] for item in batch]
    collated['risk_id'] = [item['risk_id'] for item in batch]
    
    return collated
