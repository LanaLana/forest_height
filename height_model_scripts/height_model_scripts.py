import os
import cv2
import sys
import json
import random
import rasterio
import subprocess
import numpy as np
from osgeo import gdal
import tifffile as tiff
from rasterio.windows import Window

from height_model_scripts.pansharpaning import sharpen
from height_model_scripts.augmentation import augmentation 
from height_model_scripts.save_pred import split_img, reconstruct_pred


class Generator:
    def __init__(self, train_img_list, val_img_list, batch_size, num_channels, classifier_mode='regression', 
                 gdal_file_path='../height_model_scripts'):
        self.train_img_list = train_img_list
        self.val_img_list = val_img_list
        self.num_channels = num_channels
        self.IMG_ROW = 256
        self.IMG_COL = 256
        self.batch_size = batch_size
        self.img_prob = np.ones(len(train_img_list)) / len(train_img_list)
        self.augm = False
        self.sharpen_flag = False
        self.forest_mask = False
        self.clip_value = 40.
        self.lidar_coef = 5
        self.full_img=False
        self.protected_lines=False
        self.num_classes = 4
        self.classes_intervals = ((0, 4), (4, 10), (10, 20), (20, 100)) 
        self.single_img = False
        self.color_aug_prob = 0.5
        self.arctic_dem = False
        self.inference = False
        self.arctic_stat = None # {'min': , 'max': }
        
        self.sentinel = False
        
        if classifier_mode not in ['classification', 'regression', 'binary_segm']:
            print('Error! Possible modes are \'classification\', \'regression\', \'binary_segm\' ')
        else:
            self.classifier_mode = classifier_mode
        #{'classification': False, 'regression': True}
                  
        self.lidar_resolution = 1
        self.target_resolution = 1
        self.gdal_file_path = gdal_file_path
        
        #---------------------------------------------------------------------------------------
        # file names
        #---------------------------------------------------------------------------------------
        self.channels_list = ['channel_0.tif', 'channel_1.tif', 'channel_2.tif']
        self.panchromatic_name = 'pan.tif'
        self.forest_mask_name = 'class_606.tif'
        self.height_name = 'height.tif'
        self.target_name = ''
        self.arcticdem_name = []
        
        
        if 'data_gdal.json' not in os.listdir(self.gdal_file_path):
            self.write_gdalinfo_file()
            
        with open(self.gdal_file_path + '/data_gdal.json', 'r') as input_file:
            #print(self.gdal_file_path + '/data_gdal.json')
            self.data_gdal = json.load(input_file)
    
    def set_file_names(self, channels_list, height_name, panchromatic_name='', forest_mask_name=''):
        self.channels_list = [(not single_img)*'_' + channel_name + '.tif' for channel_name in channels_list]
        self.panchromatic_name = (not single_img)*'_' + panchromatic_name + '.tif'
        self.forest_mask_name = (not single_img)*'_' + forest_mask_name + '.tif'
        self.height_name = (not single_img)*'_' + height_name + '.tif'
    
    def get_img_mask_array(self, imgpath):
        num_channels=self.num_channels 
        augm=self.augm
        sharpen_flag=self.sharpen_flag
        forest_mask=self.forest_mask
        clip_value=self.clip_value
        lidar_coef=self.lidar_coef
        IMG_ROW=self.IMG_ROW
        IMG_COL=self.IMG_COL
        full_img=self.full_img
        protected_lines=self.protected_lines
        target_resolution = self.target_resolution
        lidar_resolution = self.lidar_resolution
        
        if len(imgpath.split('/')[-1]):
            self.single_img = False # if the source img is split into parts within one folder
        else:
            self.single_img = True
        
        if self.classifier_mode == 'binary_segm':
            img_name = imgpath + (not self.single_img)*'_' + self.target_name
        elif self.inference:
            img_name = imgpath + (not self.single_img)*'_' + self.channels_list[0]
        elif self.height_name == 'height.tif':
            img_name = imgpath[:-3] + 'height_' + imgpath.split('/')[-1] +'.tif' #!!!!!!!!!!!!!
            #img_name = imgpath[:-1] + 'height_' + imgpath.split('/')[-1] +'.tif'
        else:
            img_name = imgpath + (not self.single_img)*'_' + self.height_name
        with rasterio.open(img_name) as src:
            size_x = src.width
            size_y = src.height
        
        if self.full_img:
            IMG_ROW = int(size_x/(target_resolution/lidar_resolution))
            IMG_COL = int(size_y/(target_resolution/lidar_resolution))
            
            rnd_x=rnd_y=0
        else:
            rnd_x = random.randint(0,size_x -  IMG_ROW*(target_resolution//lidar_resolution) - 1)
            rnd_y = random.randint(0,size_y - IMG_COL*(target_resolution//lidar_resolution) - 1)

        window = Window(rnd_x, rnd_y, int(IMG_ROW*(target_resolution/lidar_resolution)), #//
                        int(IMG_COL*(target_resolution/lidar_resolution))) #//

        img = np.zeros((IMG_COL, IMG_ROW, num_channels+sharpen_flag), dtype=np.uint8)#
        with rasterio.open(img_name) as src:
            mask_0 = src.read(window=window).astype(np.float)
        #---------------------------------------------------------------------------------------
        # protected lines
        #---------------------------------------------------------------------------------------
        if protected_lines:
            try:
                window_x0, window_y0, window_x1, window_y1  = self.extract_coord(imgpath, rnd_x, rnd_y, IMG_ROW, IMG_COL)
                if imgpath.split('/')[-1] + '_line.tif' in os.listdir(imgpath[:-3]):
                    lines_mask = (self.channel_resize(imgpath, '_line.tif', window_x0, window_y0, 
                                                window_x1, window_y1, IMG_ROW, IMG_COL)>0).astype(np.uint8)
                mask_0*=lines_mask
            except:
                img,mask=self.get_img_mask_array(imgpath)     
        
        #---------------------------------------------------------------------------------------
        # remove black area (areas without height data covering)
        #---------------------------------------------------------------------------------------
        while (np.sum(np.where(mask_0>0, 1, 0)) < IMG_ROW*IMG_COL*3/5.) and not full_img \
            and not (self.classifier_mode == 'binary_segm'):
            rnd_x = random.randint(0,size_x -  IMG_ROW*(target_resolution//lidar_resolution) - 1)
            rnd_y = random.randint(0,size_y - IMG_COL*(target_resolution//lidar_resolution) - 1)
            window = Window(rnd_x, rnd_y, IMG_ROW*(target_resolution//lidar_resolution), 
                        IMG_COL*(target_resolution//lidar_resolution))
            with rasterio.open(img_name) as src:
                mask_0 = src.read(window=window).astype(np.float)
            #---------------------------------------------------------------------------------------
            # protected lines
            #---------------------------------------------------------------------------------------
            if protected_lines:
                try:
                    window_x0, window_y0, window_x1, window_y1  = self.extract_coord(imgpath, rnd_x, rnd_y, IMG_ROW, IMG_COL)
                    if imgpath.split('/')[-1] + '_line.tif' in os.listdir(imgpath[:-3]):
                        lines_mask = (self.channel_resize(imgpath, '_line.tif', window_x0, window_y0, 
                                                    window_x1, window_y1, IMG_ROW, IMG_COL)>0).astype(np.uint8)
                    mask_0*=lines_mask
                except:
                    img,mask=self.get_img_mask_array(imgpath)
                
        
        if target_resolution != lidar_resolution:
            mask_0 = np.expand_dims(cv2.resize(mask_0[0], (IMG_ROW, IMG_COL), interpolation=cv2.INTER_NEAREST), 0)
            
        #---------------------------------------------------------------------------------------
        # downsample height data
        #---------------------------------------------------------------------------------------
        if lidar_coef: # downsampling coef
            mask_0 = np.expand_dims(cv2.resize(mask_0[0], (int(IMG_ROW/lidar_coef), int(IMG_COL/lidar_coef)), interpolation=cv2.INTER_AREA), 0)
            mask_0 = np.expand_dims(cv2.resize(mask_0[0], (IMG_ROW, IMG_COL), interpolation=cv2.INTER_NEAREST), 0)
            
        #---------------------------------------------------------------------------------------
        # extract coordinates corresponding to the window of the height img
        #---------------------------------------------------------------------------------------
        window_x0, window_y0, window_x1, window_y1  = self.extract_coord(imgpath, rnd_x, rnd_y, img_name,
                                                                         int(IMG_ROW*(target_resolution/lidar_resolution)), 
                                                                         int(IMG_COL*(target_resolution/lidar_resolution)))

        #---------------------------------------------------------------------------------------
        # clean data
        #---------------------------------------------------------------------------------------
        if forest_mask:
            channel_name = (not self.single_img)*'_' + self.forest_mask_name
            mask_606 = self.channel_resize(imgpath, channel_name, window_x0, window_y0, 
                                      window_x1, window_y1, IMG_ROW, IMG_COL)

            felling = np.where(mask_0 < 1., 1, 0) * mask_606
            cloud = np.where(mask_0 > 5., 1, 0) * np.where(mask_606>0, 0, 1)
            mask_0 *= np.where(felling==1, 0, 1) * np.where(cloud==1, 0, 1)

        #---------------------------------------------------------------------------------------
        # read img bands
        #---------------------------------------------------------------------------------------
        for i, channel_name in enumerate(self.channels_list):
            channel_name = (not self.single_img)*'_' + channel_name 
            img[:,:,i] = self.channel_resize(imgpath, channel_name, window_x0, window_y0, 
                                        window_x1, window_y1, IMG_ROW, IMG_COL) 
            
        mask = np.ones((img.shape[0], img.shape[1], 1)) 
        mask[:,:,0] = mask_0 
        
        #---------------------------------------------------------------------------------------
        # read Arctic DEM
        #---------------------------------------------------------------------------------------
        if self.arctic_dem:
            if self.sentinel:
                channel_name = (not self.single_img)*'_' + 'arcticdem.tif' 
                #channel_name = '/home/user/data/projects/research-project/notebooks/Illarionova/usgs_sentinel/Left_shore/krasnoborsk/arctic_dem.tif'
            else:
                channel_name = (not self.single_img)*'_' + 'arcticdem.tif' 
            img[:,:,len(self.channels_list)] = self.channel_resize(imgpath, channel_name, window_x0, window_y0, 
                                        window_x1, window_y1, IMG_ROW, IMG_COL) 
            #img[:,:,len(self.channels_list)] = 255. * (img[:,:,len(self.channels_list)] - self.arctic_stat['min']) / \
            #    (self.arctic_stat['max'] - self.arctic_stat['min'])
            #tiff.imshow(img[:,:,len(self.channels_list)])
        
        #---------------------------------------------------------------------------------------
        # pansharpaning
        #---------------------------------------------------------------------------------------
        if sharpen_flag:
            channel_name = (not self.single_img)*'_' + self.panchromatic_name
            img[:,:,-1] = self.channel_resize(imgpath, channel_name, window_x0, window_y0, 
                                         window_x1, window_y1, IMG_ROW, IMG_COL) 
            #return img[:,:,-1]
            if not img[:,:,-1].std(): # check data validity
                #print('something wrong')
                img,mask=self.get_img_mask_array(imgpath)
                return img, mask

        #---------------------------------------------------------------------------------------
        # augmentation
        #---------------------------------------------------------------------------------------
        if augm:
            # color augmentation is implemented just for RGB img
            self.color_aug_prob *= (3 == len(self.channels_list)) 
            img, mask_tmp  = augmentation(img, mask, self.color_aug_prob)
            if len(mask_tmp.shape)==2:
                mask[:,:,0]=mask_tmp
            else:
                mask=mask_tmp
        
        img, mask = np.asarray(img / 255.), np.asarray(mask)

        if sharpen_flag:
            sharp = sharpen(img[:,:,-1], np.asarray([img[:,:,0], img[:,:,1], img[:,:,2]]))
            for i in range(3):
                img[:,:,i] = sharp[i,:,:]

        #---------------------------------------------------------------------------------------
        # check data validity (will be removed)
        #---------------------------------------------------------------------------------------
        if self.classifier_mode == 'binary_segm':
            pass
        elif np.sum(img[:,:,:-1])==0 or np.sum(mask.clip(0, 30) / 30.)==0 \
            or np.nan in img[:,:,:-1] or np.nan in mask.clip(0, 30) / 30. \
            or np.inf in img[:,:,:-1] or np.inf in mask.clip(0, 30) / 30.:
            print('wrong input loss ', imgpath)
            img,mask=self.get_img_mask_array(imgpath)
            return img, mask
        
        #---------------------------------------------------------------------------------------
        # specify classification task
        #---------------------------------------------------------------------------------------
        if self.classifier_mode == 'classification':
            mask = self.set_classification_mask(mask[:,:,0])
        elif self.classifier_mode == 'regression':
            mask = mask.clip(0, clip_value) / clip_value
        elif self.classifier_mode == 'binary_segm':
            pass
        else:
            print('error! set classification mode')
            return 0
            
        return img[:,:,:num_channels], mask

    def set_classification_mask(self, mask):
        classification_mask = np.ones((mask.shape[0], mask.shape[1], self.num_classes), dtype=np.uint8)
        for cl, interval in enumerate(self.classes_intervals):
            classification_mask[:,:,cl] = (mask >= interval[0]) * (mask < interval[1])
        classification_mask[:,:,-1] = mask >= interval[0]
        return classification_mask
        
    def train_gen(self):
        while(True):
            imgarr=[]
            maskarr=[]
            train_samples = np.random.choice(self.train_img_list, self.batch_size, p=self.img_prob)
            for i in range(self.batch_size):
                #print(i)
                img,mask=self.get_img_mask_array(train_samples[i])
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
                rnd_id=random.randint(0,len(self.val_img_list)-1)
                img,mask=self.get_img_mask_array(self.val_img_list[rnd_id])

                imgarr.append(img)
                maskarr.append(mask)

            yield (np.asarray(imgarr),np.asarray(maskarr))
            imgarr=[]
            maskarr=[]
    
    def extract_coord(self, imgpath, rnd_x, rnd_y, img_name, IMG_ROW=256, IMG_COL=256):
        #--------------------------------------------------------------
        # panchrom
        #--------------------------------------------------------------
        
        key = img_name
        
        dx = self.data_gdal[key]['dx']
        dy = self.data_gdal[key]['dy']

        x0 = self.data_gdal[key]['x0']
        y0 = self.data_gdal[key]['y0']

        size_x = self.data_gdal[key]['size_x']
        size_y = self.data_gdal[key]['size_y']

        x1 = x0 + size_x * dx
        y1 = y0 + size_y * dy

        window_x0 = x0 + rnd_x * dx
        window_y0 = y0 + rnd_y * dy

        window_x1 = window_x0 + IMG_ROW * dx
        window_y1 = window_y0 + IMG_COL * dy

        return window_x0, window_y0, window_x1, window_y1

    def channel_resize(self, imgpath, channel_name, window_x0, window_y0, window_x1, window_y1, IMG_ROW=256, IMG_COL=256):
        #--------------------------------------------------------------
        # other channels
        #--------------------------------------------------------------
        
        if 'arctic' in channel_name and self.sentinel:
            key = '/home/user/data/projects/research-project/notebooks/Illarionova/usgs_sentinel/Left_shore/krasnoborsk/arctic_dem.tif'
        else:
            key = imgpath + channel_name
        
        dx = self.data_gdal[key]['dx']
        dy = self.data_gdal[key]['dy']

        x0 = self.data_gdal[key]['x0']
        y0 = self.data_gdal[key]['y0']

        size_x = self.data_gdal[key]['size_x']
        size_y = self.data_gdal[key]['size_y']

        coord_x0 = abs((x0 - window_x0)/dx)
        coord_y0 = abs((y0 - window_y0)/dy)

        coord_x1 = abs((x0 - window_x1)/dx) 
        coord_y1 = abs((y0 - window_y1)/dy) 

        window = Window(coord_x0, coord_y0, coord_x1 - coord_x0, coord_y1 - coord_y0)
        
        with rasterio.open(key) as src:#imgpath + channel_name
            img_crop = src.read(window=window)

        dim = (IMG_ROW, IMG_COL) 
        new_crop = img_crop.transpose(1,2,0)
        new_crop = np.expand_dims(cv2.resize(new_crop, dim, interpolation=cv2.INTER_NEAREST), 0)

        return new_crop

    def set_prob(self):
        #--------------------------------------------------------------
        # function to set distribution
        #--------------------------------------------------------------
        # set probability for each DSE element as a partion of  
        # region square in comparasion with other pathces
        #
        
        print('Not implemented')
        '''
        img_prob = np.zeros((len(self.train_img_list)))
        for i, img_path in enumerate(self.train_img_list):
            for cl in self.class_0+self.class_1:
                if cl+'_05.tif' in os.listdir(img_path):
                    img_prob[i] += np.sum(tiff.imread(img_path+'/'+cl+'_05.tif'))
        img_prob = img_prob/np.sum(img_prob)
        return img_prob
        '''
    
    def pred_img(self, model, img_path, num_classes=1, inference = False, overlap = 128, crop_size = 512): 
        #--------------------------------------------------------------
        # pred_img function allows to make prediction for an image of 
        # any size (not the power of 2)
        #--------------------------------------------------------------
        
        IMG_ROW = IMG_COL = crop_size
        
        self.full_img = True
        self.inference = inference
        img,mask = self.get_img_mask_array(img_path)
        self.inference = False
        self.full_img = False
        
        size_x, size_y, _ = img.shape

        imgarr, height_ind, width_ind = split_img(img, IMG_ROW, IMG_COL, overlap)
        
        pred = np.asarray([model.predict(np.asarray([img_patch]))[0] for img_patch in imgarr])
        recon = reconstruct_pred(pred, size_x, size_y, IMG_ROW, IMG_COL, overlap, height_ind, width_ind)
        
        return recon, mask
    
    def write_gdalinfo_file(self):
        data = {}
        #print('f')
        for imgpath in self.train_img_list + self.val_img_list:
            #print(self.train_img_list + self.val_img_list)
            bands_list = [imgpath + (not self.single_img)*'_' + channel_name for channel_name in self.channels_list]
            #print(bands_list)
            if self.classifier_mode == 'binary_segm':
                img_name = imgpath + (not self.single_img)*'_' + self.target_name
            elif self.inference:
                img_name = imgpath + (not self.single_img)*'_' + self.channels_list[0]
            elif self.height_name == 'height.tif':
                img_name = imgpath[:-3] + 'height_' + imgpath.split('/')[-1] +'.tif' #!!!!!!!!!!
                #img_name = imgpath[:-1] + 'height_' + imgpath.split('/')[-1] +'.tif'
            else:
                img_name = imgpath + (not self.single_img)*'_' + self.height_name
            bands_list += [img_name]
            if self.forest_mask:
                bands_list += [imgpath + (not self.single_img)*'_' + self.forest_mask_name] 
            if self.sharpen_flag:
                bands_list += [imgpath + (not self.single_img)*'_' + self.panchromatic_name] 
                #self.forest_mask_name self.height_name self.panchromatic_name
            if self.arctic_dem:
                if self.sentinel:
                    bands_list += ['/home/user/data/projects/research-project/notebooks/Illarionova/usgs_sentinel/Left_shore/krasnoborsk/arctic_dem.tif']
                else:
                    bands_list += [imgpath + (not self.single_img)*'_' + 'arcticdem.tif'] 
            #print(bands_list)
            for key in bands_list:
                if key.split('/')[-1] not in os.listdir(key[:-len(key.split('/')[-1])]):
                    print(key.split('/')[-1] +' what?')
                    #print(key.split('/')[-1], os.listdir(imgpath[:-len(key.split('/')[-1])]))
                    continue
                if key not in data.keys():
                    data[key] = {}
                    
                ret = str(subprocess.check_output("gdalinfo {}".format(key), shell=True))
                
                n = ret.find('Pixel Size')+len('Pixel Size = (')
                data[key]['dx'] = float(ret[n:n+ret[n:].find(',')])
                data[key]['dy'] = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

                n = ret.find('Origin = (') + len('Origin = (')
                data[key]['x0'] = float(ret[n:n+ret[n:].find(',')])
                data[key]['y0'] = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

                n = ret.find('Size is ')+len('Size is ')
                data[key]['size_x'] = int(ret[n:n+ret[n:].find(',')])
                data[key]['size_y'] = int(ret[2+n+ret[n:].find(','):n+ret[n:].find('\\')])
           
        with open(self.gdal_file_path + '/data_gdal.json', 'w') as outfile:
            json.dump(data, outfile)
            
        self.data_gdal = data
    