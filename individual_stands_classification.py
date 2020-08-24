import os
import cv2
import sys
import math
import scipy
import random
import rasterio
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import seed
from glob import glob
from rasterio.windows import Window

from keras.models import model_from_json
from keras import backend as K
from keras.layers import Conv2D
from keras import layers
from keras.models import Model
import tensorflow as tf
import random
import json

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    weights_list = np.zeros((len(keys)))
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        weights_list[sorted(keys).index(key)] = class_weight[key]
    return class_weight, weights_list

class Generator:
    def __init__(self, batch_size, class_0, class_1, num_channels):
        self.num_channels = num_channels
        self.num_classes = 2
        self.IMG_ROW = 64
        self.IMG_COL = 64
        self.batch_size = batch_size
        self.class_0 = class_0
        self.class_1 = class_1
        self.normalize_channel = None
        self.sup_materials = False
        self.sup_name = None
        self.sup_normalization = None
        self.sentinel = False
        self.val_region = False
        self.wv = False
        
        self.channels_name = ['B02','B03','B04','B05','B06','B07','B08','B11','B12','B8A']
        
        self.json_file_linden_val = None
        self.json_file_oak_val = None
        self.json_file_linden_train = None
        self.json_file_oak_train = None
    
    def set_normalize_channel(self):
        self.normalize_channel = {}
        for path in ['/home/user/data/projects/research-project/notebooks/Illarionova/Forestry_inventory/wv_inference/krasnoborsk/0','/home/user/data/projects/research-project/notebooks/Illarionova/Forestry_inventory/wv_inference/krasnoborsk/1',
                    '/home/user/data/projects/research-project/notebooks/Illarionova/Forestry_inventory/wv_inference/krasnoborsk/2','/home/user/data/projects/research-project/notebooks/Illarionova/Forestry_inventory/wv_inference/krasnoborsk/3']:
            self.normalize_channel[path] = {}
            tmp = []
            for ch in range(8):
                with rasterio.open(path + '_channel_' + str(ch) + '.tif') as src:
                    tmp += [src.read(1)]
            tmp = np.asarray(tmp)
            self.normalize_channel[path] = []
            self.normalize_channel[path] += [np.mean(tmp[tmp>0])]
            self.normalize_channel[path] += [np.std(tmp[tmp>0])]
            self.normalize_channel[path] += [tmp.max()]
            self.normalize_channel[path] += [tmp[tmp>0].min()] #tmp.min

    def get_img_mask_array(self, imgpath, upper_left_x, upper_left_y, pol_width, pol_height, age_flag = False):
        #class_0=['S'], class_1=['E']
        #print(imgpath)
        if self.wv:
            channel_name = imgpath + self.channels_name[0] + '.tif'
        else:
            channel_name = '_'.join(imgpath.split('_')[:-1]) + self.channels_name[0] + '.tif'
        
        with rasterio.open(channel_name) as src:
            size_x = src.width
            size_y = src.height
        #print(upper_left_x, upper_left_y, pol_width, pol_height)
        difference_x = max(0, self.IMG_COL - int(pol_width))
        difference_y = max(0, self.IMG_ROW - int(pol_height))
        rnd_x = random.randint(max(0, int(upper_left_x) - difference_x),min(size_x, 
                                                     int(upper_left_x) + int(pol_width) + difference_x) -
                              self.IMG_COL)
        rnd_y = random.randint(max(0, int(upper_left_y) - difference_y),min(size_y, 
                                                     int(upper_left_y) + int(pol_height) + difference_y) -
                              self.IMG_ROW)
        
        window = Window(rnd_x, rnd_y, self.IMG_COL, self.IMG_ROW)
        
        mask_0 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
        for cl_name in self.class_0:
            #if '{}.tif'.format(cl_name) in os.listdir(imgpath):
            if self.wv:
                channel_name = imgpath + '_'+ cl_name + '.tif'
            else:
                channel_name = '_'.join(imgpath.split('_')[:-1]) +(not self.sentinel)*'_'+ '{}.tif'.format(cl_name)

            with rasterio.open(channel_name) as src:
                mask_0 += src.read(window=window).astype(np.int)

        mask_1 = np.zeros((1, self.IMG_ROW, self.IMG_COL))
        for cl_name in self.class_1:
            #if '{}.tif'.format(cl_name) in os.listdir(imgpath):
            if self.wv:
                channel_name = imgpath +'_'+ cl_name + '.tif'
            else:
                channel_name = '_'.join(imgpath.split('_')[:-1]) +(not self.sentinel)*'_'+ '{}.tif'.format(cl_name)

            with rasterio.open(channel_name) as src:
                mask_1 += src.read(window=window).astype(np.int)
        #mask_1 = mask_1 > 0.5

        img = np.ones((self.IMG_ROW, self.IMG_COL, self.num_channels), dtype=np.float)
        for i, ch in enumerate(self.channels_name):
            if self.wv:
                channel_name = imgpath + ch + '.tif'
            else:
                channel_name = '_'.join(imgpath.split('_')[:-1])+ch+ '.tif'

            with rasterio.open(channel_name) as src:
                img[:,:,i] = src.read(window=window)
        if self.normalize_channel != None:
            width=3
            mean = self.normalize_channel[imgpath][0]
            std = self.normalize_channel[imgpath][1]
            img_max = self.normalize_channel[imgpath][2]
            img_min = self.normalize_channel[imgpath][3]
            m = max(0, mean - width*std)
            M = min(img_max, mean + width*std)

            img = ((img - m)/(M-m)).clip(0., 1.)
        else:
            img /= 255.
        img = img.clip(0, 1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # suplementary materials
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.sup_materials:
            if self.wv:
                channel_name = imgpath + self.sup_name
            else:
                channel_name = '_'.join(imgpath.split('_')[:-1]) + self.sup_name
            with rasterio.open(channel_name) as src:
                img[:,:,-1] = src.read(window=window).astype(np.float)
            img[:,:,-1] = (img[:,:,-1] / self.sup_normalization).clip(0., 1.)

        mask = np.ones((self.IMG_ROW, self.IMG_COL, self.num_classes)) 
        mask[:,:,0] = mask_0  
        mask[:,:,1] = mask_1 

        return np.asarray(img), np.asarray(mask)  
    
    def extract_val(self, sample):
        return sample['upper_left_x'], sample['upper_left_y'], sample['pol_width'], sample['pol_height']
    
    def train_gen(self):
        while(True):
            imgarr=[]
            maskarr=[]
            for i in range(self.batch_size):
                if random.random() > 0.5:
                    random_key = random.choice(list(self.json_file_oak_train.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_oak_train[random_key])
                else:
                    random_key = random.choice(list(self.json_file_linden_train.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_linden_train[random_key])
                if not self.sentinel:
                    img_name = '/home/user/data/projects/research-project/notebooks/Illarionova/Forestry_inventory/wv_inference/krasnoborsk/'+random_key.split('_')[0]
                else:
                    img_name = random_key#.split('_')[0]
                    #print(img_name)
                img,mask=self.get_img_mask_array(img_name, upper_left_x, upper_left_y, pol_width, pol_height)
                imgarr.append(img)
                maskarr.append(mask)
            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[] 

    def val_gen(self):
        while(True):
            imgarr=[]
            maskarr=[]
            for i in range(self.batch_size):
                if random.random() > 0.5:
                    random_key = random.choice(list(self.json_file_oak_val.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_oak_val[random_key])
                else:
                    random_key = random.choice(list(self.json_file_linden_val.keys()))
                    upper_left_x, upper_left_y, pol_width, pol_height = self.extract_val(self.json_file_linden_val[random_key])
                if not self.sentinel:
                    img_name = '/home/user/data/projects/research-project/notebooks/Illarionova/Forestry_inventory/wv_inference/krasnoborsk/'+random_key.split('_')[0]
                else:
                    img_name = random_key#.split('_')[0]
                img,mask=self.get_img_mask_array(img_name, upper_left_x, upper_left_y, pol_width, pol_height)
                imgarr.append(img)
                maskarr.append(mask)
            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[]
            
    def set_prob(self):
        img_prob = np.zeros((len(self.train_img_list)))
        for i, img_path in enumerate(self.train_img_list):
            for cl in self.class_0+self.class_1:
                if cl+'_05.tif' in os.listdir(img_path):
                    img_prob[i] += np.sum(tiff.imread(img_path+'/'+cl+'_05.tif'))
        img_prob = img_prob/np.sum(img_prob)
        return img_prob

    def weighted_categorical_crossentropy(self, weights):
        def loss(target,output,from_logits=False):
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            non_zero_pixels = tf.reduce_sum(target, axis=-1)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1) \
                    * (self.IMG_ROW*self.IMG_COL*self.batch_size) / K.sum(non_zero_pixels)

        return loss
    
    def read_json(self, folders, class_name):
        js_full = {}
        samples_set = set()
        for folder in folders:
            if self.wv:
                json_file = '/home/user/data/projects/research-project/notebooks/Illarionova/Forestry_inventory/wv_inference/krasnoborsk/{}_{}.json'.format(folder, class_name) # folder 0 1
            else:
                json_file = folder + class_name + '.json'
            with open(json_file, 'r') as f:
                js_tmp = json.load(f)
            keys_list = set(js_tmp.keys())
            for key in keys_list:
                #if tuple(self.extract_val(js_tmp[key])) not in samples_set:
                js_tmp[folder+'_'+key] = js_tmp[key]
                samples_set.add(tuple(self.extract_val(js_tmp[key])))
                del js_tmp[key]
            js_full.update(js_tmp)
        return js_full
    
    def train_val_split(self, json_file, split_ration):
        keys_list = set(json_file.keys())
        train_samples, val_samples = {}, {}
        if self.val_region:
            with open('val_region.json', 'r') as f:
                val_region_dict = json.load(f) 
            for key in keys_list:
                val_region_flag = False
                for ind in val_region_dict.keys():
                    if json_file[key]['upper_left_x'] < val_region_dict[ind]['upper_left_x'] + \
                        val_region_dict[ind]['pol_width']\
                        and json_file[key]['upper_left_x'] > val_region_dict[ind]['upper_left_x'] \
                        and json_file[key]['upper_left_y'] < val_region_dict[ind]['upper_left_y'] + \
                        val_region_dict[ind]['pol_height']\
                        and json_file[key]['upper_left_y'] > val_region_dict[ind]['upper_left_y']:
                        val_region_flag = True
                if val_region_flag:
                    val_samples[key] = json_file[key]
                else:
                    train_samples[key] = json_file[key]
        else:
            seed(1)
            for key in keys_list:
                if random.random() < split_ration:
                    train_samples[key] = json_file[key]
                else:
                    val_samples[key] = json_file[key]
        return train_samples, val_samples

    def load_dataset(self, folders, split_ration=0.7):
        json_file_oak_train = self.read_json(folders, "conifer")
        json_file_linden_train = self.read_json(folders, "decidious")
        
        self.json_file_oak_train, self.json_file_oak_val = self.train_val_split(json_file_oak_train, split_ration)
        self.json_file_linden_train, self.json_file_linden_val = self.train_val_split(json_file_linden_train, split_ration)