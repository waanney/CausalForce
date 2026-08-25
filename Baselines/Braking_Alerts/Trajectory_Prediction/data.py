import os
import numpy as np
import torch 
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class MultipleRisksDataset(Dataset):
    
    def __init__(self, data_root='./YOUR_DATA_ROOT_PATH'):
        self.data_root = data_root
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.front_imgs = []
        self.labels = []
        self.risk_ids = []
        self.objects_list = []
        self.target_img = []
        self.scenario_id_list = []
        self.risk_interval_H8_list = []
        self.traj_H8_normalized_list = []
        self.actor_data_list = []
        self.measurement_list = []

        for scenario in os.listdir(data_root):
            images = sorted(os.listdir(self.data_root+scenario+'/rgb_front/'))
            
            for i in range(0, len(images)-2):
                front_imgs = []
                objects = []
                for j in range(i, i+3):
                    front_imgs.append(self.data_root+scenario+'/rgb_front/'+images[j])
                    objects.append(self.data_root+scenario+'/2d_bbs_front/'+images[j][:4]+'.npy')
                self.front_imgs.append(front_imgs)
                self.objects_list.append(objects)
                
                self.labels.append(self.data_root+scenario+'/2d_bbs_front/'+images[i+2][:4]+'.npy')
                self.risk_interval_H8_list.append(self.data_root+scenario+'/risk_interval_H8_new/'+images[i+2][:4]+'.npy')
                self.traj_H8_normalized_list.append(self.data_root+scenario+'/trajectory_H8_normalized/'+images[i+2][:4]+'.npy')
                self.target_img.append(self.data_root+scenario+'/rgb_front/'+images[i+2])
                self.risk_ids.append(self.data_root+scenario+'/risk_id.json')
                self.scenario_id_list.append(scenario)
                self.actor_data_list.append(self.data_root+scenario+'/actors_data/'+images[i+2][:4]+'.json')
                self.measurement_list.append(self.data_root+scenario+'/measurements/'+images[i+2][:4]+'.json')

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
        all_obj_dist = []
        with open(self.actor_data_list[index]) as f:
            actors = json.load(f)
        ego_id = next(iter(actors))
        ego_xy  = actors[str(ego_id)]["loc"][:2]

        for id in all_objects_id[2]:
            if str(id) not in actors.keys():
                all_obj_dist.append(1000.0)
                continue
            obj_xy = actors[str(id)]["loc"][:2]
            dist = np.linalg.norm(np.array(ego_xy)-np.array(obj_xy))
            all_obj_dist.append(dist)
       
        with open(self.measurement_list[index]) as f:
            measurement = json.load(f)
            if measurement['brake']:
                brake = 1
            else:
                brake = 0
        
        risk_interval_H8 = np.load(self.risk_interval_H8_list[index], allow_pickle=True)
        risk_ids = [int(key) for key in risk_interval_H8.item().keys()]
        risk_interval_H8_tensor = [torch.tensor(risk_interval_H8.item()[str(item)], dtype=torch.float) for item in risk_ids]   

        trajctory_H8_normalized = np.load(self.traj_H8_normalized_list[index], allow_pickle=True)
        all_objects_id_curr = [int(key) for key in trajctory_H8_normalized.item().keys()]
        trajctory_H8_normalized_tensor = [torch.tensor(trajctory_H8_normalized.item()[item], dtype=torch.float) for item in all_objects_id_curr]

        data['target_img'] = Image.open(self.target_img[index]).convert("RGB")
        data['front_imgs'] = input_tensor
        data['all_objs_bbs'] = all_objects
        data['scenario_id'] = self.scenario_id_list[index]
        data['risk_interval_H8'] = risk_interval_H8_tensor
        data['traj_H8_normalized'] = trajctory_H8_normalized_tensor
        data['risk_id'] = risk_ids
        data['all_objs_id'] = all_objects_id
        data['all_objects_id_curr'] = all_objects_id_curr
        data["all_obj_dist"] = all_obj_dist
        data['brake'] = brake

        return data

    def get_all_objects(self, bbs_2d_front):
        #print(bbs_2d_front)
        all_objs = []
        all_ids = []
        for id, bb in bbs_2d_front.item()['obstacles']:
            all_objs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])
            all_ids.append(id)
        for id, bb in bbs_2d_front.item()['vehicles']:
            all_objs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])
            all_ids.append(id)
        for id, bb in bbs_2d_front.item()['pedestrians']:
            all_objs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])   
            all_ids.append(id)
        if 'phantom_objects' in bbs_2d_front.item().keys():
            for id, bb in bbs_2d_front.item()['phantom_objects']:
                all_objs.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])   
                all_ids.append(id)

        return all_objs, all_ids
    

def custom_collate_fn(batch):
    collated = {}
    
    collated['front_imgs'] = torch.stack([item['front_imgs'] for item in batch], dim=0)
    collated['all_objs_bbs'] = [item['all_objs_bbs'] for item in batch]
    collated['all_objs_id'] = [item['all_objs_id'] for item in batch]
    collated['target_img'] = [item['target_img'] for item in batch]
    collated['scenario_id'] = [item['scenario_id'] for item in batch]
    collated['risk_interval_H8'] = [item['risk_interval_H8'] for item in batch]
    collated['traj_H8_normalized'] = [item['traj_H8_normalized'] for item in batch]
    collated['risk_id'] = [item['risk_id'] for item in batch]
    collated['all_objects_id_curr'] = [item['all_objects_id_curr'] for item in batch]
    collated['all_obj_dist'] = [item['all_obj_dist'] for item in batch]
    collated['brake'] = [item['brake'] for item in batch]
    
    return collated
    