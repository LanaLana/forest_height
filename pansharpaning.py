import cv2
import numpy as np

def sharpen(pan, ms, normalize=True):
    """
    :param pan:
    :param ms:
    :return:
    """
    # The methods expect already resized images

    assert pan.shape == ms.shape[1:], 'Shapes of multispectral and panchrom are different'
    assert ms.shape[0] == 3, 'IHS pansharpening is restricted to 3 channels, use GIHS for others'
    ms = ms.transpose(1, 2, 0)

    pan = pan.astype(np.float32)
    ms = ms.astype(np.float32)

    hsv = cv2.cvtColor(ms, cv2.COLOR_RGB2HSV)

    #pan = normalize_meanstd(base=hsv[:, :, 3], channel=pan) #

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:,:,2]

    if normalize:
        pan = normalize_meanstd(value, pan)

    HSpan = np.array([hue, saturation, pan]).transpose(1, 2, 0)
    RGBpan = cv2.cvtColor(HSpan, cv2.COLOR_HSV2RGB)

    return RGBpan.transpose(2, 0, 1)

def saturate_cast(img, dtype):
    """
    Implementation of opencv saturete_cast, which chenges datatype and clips the data to the output type range
    :param img: input image
    :param dtype: data type of output image
    :return: new image of dtype
    """
    if np.issubdtype(dtype, np.integer):
        return np.clip(img, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    else:
        return img.astype(dtype)

def normalize_meanstd(base, channel, mask=None):
    if mask is None:
        mask = np.ones(shape=base.shape, dtype=np.uint8) > 0
    else:
        mask = mask > 0
        
    base_mean = base[mask].mean()
    
    channel_ = channel.astype(np.float32)
    if channel[mask].std():
        channel_ = (channel_ - channel[mask].mean())/channel[mask].std() # !!!!!!
    else:
        print('div 0')
        channel_ = (channel_ - channel[mask].mean())
    channel_ = channel_*base[mask].std() + base[mask].mean()
    return channel_