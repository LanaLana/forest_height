import subprocess
import numpy as np
from osgeo import gdal, osr

from aeronet.dataset import polygonize, BandCollection, FeatureCollection, parse_directory

def save_pred(img_to_save, save_dist, info_img, crs=3857):  
    #---------------------------------------------------------------------------------------
    # img_to_save: binary np.array with 2 dim
    # save_dist: name with path
    # info_img: image to extract geo coordinates for the img_to_save (one of RGB bands)
    #---------------------------------------------------------------------------------------
    ret = str(subprocess.check_output("gdalinfo {}".format(info_img), shell=True))
     
    n = ret.find('Pixel Size')+len('Pixel Size = (')
    dx = float(ret[n:n+ret[n:].find(',')])
    dy = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

    n = ret.find('Origin = (') + len('Origin = (')
    xmin = float(ret[n:n+ret[n:].find(',')])
    ymax = float(ret[1+n+ret[n:].find(','):n+ret[n:].find(')')])

    n = ret.find('Size is ')+len('Size is ')
    size_x = int(ret[n:n+ret[n:].find(',')])
    size_y = int(ret[2+n+ret[n:].find(','):n+ret[n:].find('\\')])
    
    xmax = xmin + size_x * dx
    ymin = ymax + size_y * dy

    xres = (xmax - xmin) / img_to_save.shape[1] #dx
    yres = (ymax - ymin) / img_to_save.shape[0] #dy
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    #dst_ds = gdal.GetDriverByName('GTiff').Create(save_dist, size_x, size_y, 1, gdal.GDT_Byte)
    dst_ds = gdal.GetDriverByName('GTiff').Create(save_dist, img_to_save.shape[1], img_to_save.shape[0], 1, gdal.GDT_Byte)
    
    print(size_x, size_y, img_to_save.shape)
    img_to_save = (img_to_save).astype(np.uint8)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(crs)                 #  32639 WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(img_to_save)   # write r-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None
    
def tif2geojson(source_dir, tif_name, geojson_path):
    #---------------------------------------------------------------------------------------
    # source_dir: dir with tif file
    # tif_name: name without '.tif' extantion
    # geojson_path: path with target name of file
    #---------------------------------------------------------------------------------------
    mask = BandCollection(parse_directory(source_dir, [tif_name]))
    vector = polygonize(mask[0])
    vector.save(geojson_path)
