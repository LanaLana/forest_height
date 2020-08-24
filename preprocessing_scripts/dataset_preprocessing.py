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
from shapely.geometry import Polygon, box

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
    #imgpath = imgpath[:-10]+'02_10m.tif'
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
    
    return window_x0, window_y0, window_x1, window_y1, size_x, size_y, dx, dy

def band_resize(imgpath, target_size_imgpath):
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
    
    window_x0, window_y0, window_x1, window_y1, size_x_new, size_y_new, dx_new, dy_new = extract_coord(target_size_imgpath)
    
    coord_x0 = abs((x0 - window_x0)/dx_new)
    coord_y0 = abs((y0 - window_y0)/dy_new)

    coord_x1 = abs((x0 - window_x1)/dx_new) 
    coord_y1 = abs((y0 - window_y1)/dy_new) 

    with rasterio.open(imgpath) as src:
        img_crop = np.array([src.read(1)])
    
    dim = (size_x_new, size_y_new) 
    new_crop = img_crop.transpose(1,2,0)
    new_crop = np.expand_dims(cv2.resize(new_crop, dim, interpolation=cv2.INTER_NEAREST), 0)
    return new_crop

def crop_resize(input_file, target_size_imgpath, output_file, geojson_file, crs, ch=0, normalization=False, rgb_nir=False, output_type=gdal.GDT_Byte):    
    
    # create extention
    with open(geojson_file, 'r+', encoding="utf-8") as f:
        gj = geojson.load(f)
    #pol = geometry.Polygon(gj['geometry']['coordinates'][0]) #['features'][-1]
    pol = geometry.Polygon(gj['features'][-1]['geometry']['coordinates'][0])
    
    # crop tif image
    with rasterio.open(target_size_imgpath) as f: #output_file 
        chm_crop, chm_crop_affine = mask(f,[pol],crop=True)
    
    if normalization:
        if rgb_nir:
            chm_crop = from_16_to_8(chm_crop[:-1], width=3)
        else:
            chm_crop = from_16_to_8(chm_crop, width=3)
    
    chm_extent = plotting_extent(chm_crop[0], chm_crop_affine)
    geotransform = (chm_crop_affine[2], chm_crop_affine[0], 0, chm_crop_affine[5], 0,chm_crop_affine[4])
    
    new_img = band_resize(input_file, target_size_imgpath)[0]
    
    print(new_img.shape, chm_crop[0].shape[1], chm_crop[0].shape[0])
    # save new cropped image chm_crop[0]
    dst_ds = gdal.GetDriverByName('GTiff').Create(output_file,  new_img.shape[1], new_img.shape[0], 1, output_type) 
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(crs) #32638  4326          
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(new_img)   # write r-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None
    

def crop_aoi(input_file, output_file, geojson_file, output_type=gdal.GDT_Byte, crs=3857, value=1):    
    
    # export from jp2 to tif
    in_image = gdal.Open(input_file)
    driver = gdal.GetDriverByName("GTiff")
    #out_image = driver.CreateCopy(output_file, in_image, 0)
    in_image = None
    #out_image = None
    
    # create extention
    with open(geojson_file, 'r+', encoding="utf-8") as f:
        gj = geojson.load(f)
    pol = geometry.Polygon(gj['features'][-1]['geometry']['coordinates'][0])
    
    # crop tif image
    with rasterio.open(input_file, 'r+') as f: #output_file
        chm_crop, chm_crop_affine = mask(f,[pol],crop=True)
    
    chm_extent = plotting_extent(chm_crop[0], chm_crop_affine)
    geotransform = (chm_crop_affine[2], chm_crop_affine[0], 0, chm_crop_affine[5], 0,chm_crop_affine[4])
    
    #print(chm_crop.shape)
    # save new cropped image
    dst_ds = gdal.GetDriverByName('GTiff').Create(output_file,  chm_crop[0].shape[1], chm_crop[0].shape[0], 1, output_type) # gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(crs)            
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(chm_crop[0]*value)   # write r-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None
    
def read_geojson(geojson_fn):
    with open(geojson_fn, encoding="utf-8") as src:
        gj = geojson.loads(src.read())

    polys = []
    for ind, feature in enumerate(gj.features):
        if feature['geometry']['type'] == 'MultiPolygon':
            for pol in feature['geometry']['coordinates']:
                if len(pol) == 1:
                    polys.append(Polygon(pol[0], holes = None)) 
                else:
                    holes = []
                    for hole in pol[1:]:
                        holes.append(hole) 
                    polys.append(Polygon(pol[0], holes = holes)) 
        else:
            if len(feature['geometry']['coordinates']) == 1:
                    polys.append(Polygon(feature['geometry']['coordinates'][0], holes = None)) 
            else:
                holes = []
                for hole in feature['geometry']['coordinates'][1:]:
                    holes.append(hole) #[0]
                polys.append(Polygon(feature['geometry']['coordinates'][0], holes = holes)) 
    return polys

def change_crs(input_geojson, output_geojson, current_crs=4326, target_crs=3857):
    # ogr2ogr -f "GeoJSON" $dist_geojson $source_geojson -s_srs EPSG:$current_crs -t_srs EPSG:3857 
    
    with open(input_geojson, 'r+', encoding="utf-8") as f:
        gj = geojson.load(f)

    #current_crs = int(gj["crs"]["properties"]["name"].split(':')[-1])
    reprojected_list = []

    for feature in gj.features:
        if len(feature.geometry.coordinates) == 1:
            geom = feature['geometry']['coordinates'][0]
        else:
            geom = feature['geometry']['coordinates'][0]

        reprojected_geom = rasterio.warp.transform_geom(rasterio.crs.CRS.from_epsg(current_crs), 
                                                        rasterio.crs.CRS.from_epsg(target_crs), 
                                                        {'type': 'Polygon', 'coordinates': geom})
        if len(feature.geometry.coordinates) == 1:
            reprojected_list.append({'geometry':reprojected_geom})
        else:
            outer_poly = reprojected_geom
            holes = []
            for hole in feature.geometry.coordinates[1:]:
                reprojected_geom = rasterio.warp.transform_geom(rasterio.crs.CRS.from_epsg(current_crs), 
                                                                rasterio.crs.CRS.from_epsg(target_crs), 
                                                                {'type': 'Polygon', 
                                                                             'coordinates': hole})
                holes.append(reprojected_geom)
            reprojected_list.append({'geometry':{'type': 'Polygon', 
                                            'coordinates':Polygon(Polygon(outer_poly["coordinates"][0]).exterior.coords,
                                            [Polygon(inner["coordinates"][0]).exterior.coords 
                                             for inner in holes])}})

    crs = {
        "type": "name",
        "properties": {
            "name": "EPSG:" + str(target_crs)
        }
    }    
    
    feature_collection = geojson.FeatureCollection(reprojected_list, crs=crs)
    with open(output_geojson, 'w') as f:
        geojson.dump(feature_collection, f)   

def img_extention(imgpath):
    ret = str(subprocess.check_output("gdalinfo {}".format(imgpath), shell=True))

    n = ret.find('Upper Left  (  ') + len('Upper Left  (  ')
    Upper_Left_str = ret[n:n+ret[n:].find(')')]
    Upper_Left_x = float(Upper_Left_str.split(',')[0])
    Upper_Left_y = float(Upper_Left_str.split(',')[1])
    
    n = ret.find('Lower Right (  ') + len('Lower Right (  ')
    Lower_Right_str = ret[n:n+ret[n:].find(')')]
    Lower_Right_x = float(Lower_Right_str.split(',')[0])
    Lower_Right_y = float(Lower_Right_str.split(',')[1])
    
    n = ret.find('Pixel Size')+len('Pixel Size = (')
    dx = float(ret[n:n+ret[n:].find(',')])
    dy = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])
    
    return Upper_Left_x, Upper_Left_y, Lower_Right_x, Lower_Right_y, dx, dy

def img_extention2geojson(imgpath, geojson_path, current_crs):
    window_x0, window_y0, window_x1, window_y1, _, _ = img_extention(imgpath) #extract_coord(imgpath)

    geom = box(window_x0, window_y0, window_x1, window_y1)
    geom = [list(geom.exterior.coords)]

    crs = {
            "type": "name",
            "properties": {
                "name": "EPSG:" + str(current_crs)
            }
        }    
    
    feature_collection = geojson.FeatureCollection([{'geometry':{'type': 'Polygon', 'coordinates':geom}}], crs=crs)
    with open(geojson_path, 'w') as f:
        geojson.dump(feature_collection, f)

# save polygons' bounding box coordinates to json file  
def save_bbox(imgpath, polys, save_file):
    Upper_Left_x, Upper_Left_y, Lower_Right_x, Lower_Right_y, dx, dy = img_extention(imgpath)# extract img coord
    img_bbox = box(Upper_Left_x, Upper_Left_y, Lower_Right_x, Lower_Right_y)
    pol_dict = {}
    key = 0
    for pol in polys:
        if img_bbox.intersects(pol):
            #print(box(*img_bbox.intersection(pol).bounds))
            minx, miny, maxx, maxy = img_bbox.intersection(pol).bounds
            #print(img_bbox.intersection(pol).bounds) # minx, miny, maxx, maxy
            #print(Upper_Left_x, Upper_Left_y, Lower_Right_x, Lower_Right_y)
            #print(minx, miny, maxx, maxy)
            #print(Upper_Left_x-Lower_Right_x)
            #print(dx, dy)
            # translate into pixels coordinates
            pix_size = 2
            pol_left_x = int((minx - Upper_Left_x) / dx)
            pol_left_y = int((Upper_Left_y - maxy) / dx)
            pol_width = int((maxx - minx) / dx)
            pol_height = int((maxy - miny) / dx)
            
            pix_area = (img_bbox.intersection(pol).area)
            # save upper left, dx, dy
            #print(pol_left_x, pol_left_y, pol_width, pol_height)
            pol_dict[key] = {"upper_left_x": pol_left_x, "upper_left_y": pol_left_y, 
                             "pol_width": pol_width, "pol_height": pol_height, 
                             "pix_area": pix_area} # approx
            key += 1
    with open(save_file, 'w') as f:
        json.dump(pol_dict, f)

def write_tif_mask(rst_fn, out_fn, polys, polys_value=None):
    # convert geojson polygons to raster
    
    rst = rasterio.open(rst_fn)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    meta['dtype'] = rasterio.uint8 #float32
    with rasterio.open(out_fn, 'w+', **meta) as out:

        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        #shapes = (geom for geom in counties.geometry) #zip(counties.geometry, counties.LSAD_NUM))
        if polys_value != None:
            shapes = ((geom,value) for geom, value in zip(polys, polys_value))
        else:
            shapes = ((geom,1) for geom in polys)

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)
'''      
def split_into_bands(input_tif, output_tif):
    new_img = output_tif + '_channel_RED.tif'
    ! gdal_translate -b 1 $input_tif $new_img
    new_img = output_tif + '_channel_GRN.tif'
    ! gdal_translate -b 2 $input_tif $new_img
    new_img = output_tif + '_channel_BLU.tif'
    ! gdal_translate -b 3 $input_tif $new_img
'''
