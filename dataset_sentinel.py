import tifffile as tiff
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt

import geojson
from rasterio.mask import mask
from osgeo import gdal, osr
from rasterio.plot import plotting_extent

from rasterio import features
from shapely import geometry
from shapely.geometry import Polygon

import pandas as pd
import json
import re
import subprocess
from os import listdir
from pyproj import Proj, transform

import scipy
import cv2
from glob import glob

def extract_coord(imgpath):
    #--------------------------------------------------------------
    # panchrom
    #--------------------------------------------------------------
    imgpath = imgpath[:-10]+'02_10m.tif'
    ret = str(subprocess.check_output("gdalinfo {}".format(imgpath), shell=True))

    n = ret.find('Pixel Size')+len('Pixel Size = (')
    dx = float(ret[n:n+ret[n:].find(',')])
    dy = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

    n = ret.find('Origin = (') + len('Origin = (')
    x0 = float(ret[n:n+ret[n:].find(',')])
    y0 = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

    n = ret.find('Size is ')+len('Size is ')
    size_x = int(ret[n:n+ret[n:].find(',')])
    size_y = int(ret[2+n+ret[n:].find(','):n+ret[n:].find('\\')])

    x1 = x0 + size_x * dx
    y1 = y0 + size_y * dy

    window_x0 = x0
    window_y0 = y0

    window_x1 = window_x0 + x1 * dx
    window_y1 = window_y0 + y1 * dy
    
    return window_x0, window_y0, window_x1, window_y1, size_x, size_y

def band_resize(imgpath):
    #--------------------------------------------------------------
    # other channels
    #--------------------------------------------------------------

    #channel_name = '_channel_3'  
    ret = str(subprocess.check_output("gdalinfo {}".format(imgpath), shell=True))

    n = ret.find('Pixel Size')+len('Pixel Size = (')
    dx = float(ret[n:n+ret[n:].find(',')])
    dy = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

    n = ret.find('Origin = (') + len('Origin = (')
    x0 = float(ret[n:n+ret[n:].find(',')])
    y0 = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

    n = ret.find('Size is ')+len('Size is ')
    size_x = int(ret[n:n+ret[n:].find(',')])
    size_y = int(ret[2+n+ret[n:].find(','):n+ret[n:].find('\\')])
    
    window_x0, window_y0, window_x1, window_y1, size_x_new, size_y_new = extract_coord(imgpath)
    
    coord_x0 = abs((x0 - window_x0)/dx)
    coord_y0 = abs((y0 - window_y0)/dy)

    coord_x1 = abs((x0 - window_x1)/dx) 
    coord_y1 = abs((y0 - window_y1)/dy) 

    with rasterio.open(imgpath) as src:
        img_crop = np.array([src.read(1)])
    
    dim = (size_x_new, size_y_new) 
    new_crop = img_crop.transpose(1,2,0)
    new_crop = np.expand_dims(cv2.resize(new_crop, dim, interpolation=cv2.INTER_NEAREST), 0)
    return new_crop

def crop_resize(input_file, output_file, geojson_file, ch=0, normalization=False, rgb_nir=False, output_type=gdal.GDT_Byte):    
    
    # create extention
    with open(geojson_file, 'r+', encoding="utf-8") as f:
        gj = geojson.load(f)
    pol = geometry.Polygon(gj['geometry']['coordinates'][0]) #['features'][-1]
    
    # crop tif image
    with rasterio.open(output_file) as f: 
        chm_crop, chm_crop_affine = mask(f,[pol],crop=True)
    
    if normalization:
        if rgb_nir:
            chm_crop = from_16_to_8(chm_crop[:-1], width=3)
        else:
            chm_crop = from_16_to_8(chm_crop, width=3)
    
    chm_extent = plotting_extent(chm_crop[0], chm_crop_affine)
    geotransform = (chm_crop_affine[2], chm_crop_affine[0], 0, chm_crop_affine[5], 0,chm_crop_affine[4])
    
    new_img = band_resize(input_file)[0]
    
    # save new cropped image
    dst_ds = gdal.GetDriverByName('GTiff').Create(output_file,  chm_crop[0].shape[1], chm_crop[0].shape[0], 1, output_type) 
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(32638) #32638  4326          
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(new_img)   # write r-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None
    
def from_16_to_8(img16, width=3):
    mean = np.mean(img16)
    std = np.std(img16)

    m = max(0, mean - width*std)
    M = min(img16.max(), mean + width*std)

    img8 = ((img16 - m)*255.0/(M-m)).clip(1, 255).astype(np.uint8)
    return img8

#def normalization(input_file, output_file, geojson_file, normalization=False, rgb_nir=False, output_type=gdal.GDT_Byte):    
def normalization(input_file, geojson_file, normalization=False, rgb_nir=False, output_type=gdal.GDT_Byte):    
    bands_10m = ['B02','B03','B04', 'B08'] 
    bands_20m = ['B05','B06','B07','B8A','B11', 'B12'] 
    
    ## change crs
    #input_raster = gdal.Open(input_file)#output_file)
    #output_raster = output_file 
    #gdal.Warp(output_raster,input_raster,dstSRS='EPSG:4326')
    
    # create extention
    with open(geojson_file, 'r+', encoding="utf-8") as f:
        gj = geojson.load(f)
    pol = geometry.Polygon(gj['geometry']['coordinates'][0]) #['features'][-1]
    
    with rasterio.open(input_file) as src:
        size_x = src.width
        size_y = src.height
        
    chm_crop = np.zeros((len(bands_10m + bands_20m), size_y, size_x))
    # crop tif image
    for ind, band_name in enumerate(bands_10m + bands_20m):
        with rasterio.open(input_file[:-11]+band_name+'_10m.tif') as f: #output_file
            chm_crop[ind], chm_crop_affine = mask(f,[pol],crop=True)
    
    if normalization:
        chm_crop = from_16_to_8(chm_crop, width=3)

    chm_extent = plotting_extent(chm_crop[0], chm_crop_affine)
    geotransform = (chm_crop_affine[2], chm_crop_affine[0], 0, chm_crop_affine[5], 0,chm_crop_affine[4])
    
    for ind, band_name in enumerate(bands_10m + bands_20m):
        # save new cropped image
        output_file = input_file[:-11]+band_name+'_10m_norm.tif'
        dst_ds = gdal.GetDriverByName('GTiff').Create(output_file,  chm_crop[0].shape[1], chm_crop[0].shape[0], 1, output_type) 
        dst_ds.SetGeoTransform(geotransform)    # specify coords
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(32638) #  4326          
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
        dst_ds.GetRasterBand(1).WriteArray(chm_crop[ind])   # write r-band to the raster
        dst_ds.FlushCache()                     # write to disk
        dst_ds = None
        
def crop_aoi(input_file, output_file, geojson_file, ch=0, normalization=False, rgb_nir=False, output_type=gdal.GDT_Byte):    
    
    # create extention
    with open(geojson_file, 'r+', encoding="utf-8") as f:
        gj = geojson.load(f)
    pol = geometry.Polygon(gj['geometry']['coordinates'][0]) #['features'][-1]
    
    # crop tif image
    with rasterio.open(output_file) as f: 
        chm_crop, chm_crop_affine = mask(f,[pol],crop=True)
    
    if normalization:
        if rgb_nir:
            chm_crop = from_16_to_8(chm_crop[:-1], width=3)
        else:
            chm_crop = from_16_to_8(chm_crop, width=3)
    
    chm_extent = plotting_extent(chm_crop[0], chm_crop_affine)
    geotransform = (chm_crop_affine[2], chm_crop_affine[0], 0, chm_crop_affine[5], 0,chm_crop_affine[4])
    
    # save new cropped image
    dst_ds = gdal.GetDriverByName('GTiff').Create(output_file,  chm_crop[0].shape[1], chm_crop[0].shape[0], 1, output_type) 
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(32638) #32638  4326          
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(chm_crop[ch])   # write r-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None
    
def read_geojson(geojson_fn):
    with open(geojson_fn, encoding="utf-8") as src:
        gj = geojson.loads(src.read())

    polys = []
    for ind, feature in enumerate(gj.features):
        if len(feature['geometry']['coordinates']) == 1:
            polys.append(Polygon(feature['geometry']['coordinates'][0], holes = None))
        else:
            holes = []
            for hole in feature['geometry']['coordinates'][1:]:
                if len(hole) >= 3:
                    holes.append(hole)
            polys.append(Polygon(feature['geometry']['coordinates'][0], holes = holes))
    return polys

def write_tif_mask(rst_fn, out_fn, polys):
    rst = rasterio.open(rst_fn)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    meta['dtype'] = rasterio.uint8 #float32
    with rasterio.open(out_fn, 'w', **meta) as out:

        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        #shapes = (geom for geom in counties.geometry) #zip(counties.geometry, counties.LSAD_NUM))
        shapes = ((geom,1) for geom in polys)

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)