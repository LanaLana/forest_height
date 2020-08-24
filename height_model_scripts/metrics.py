import numpy as np

'''
def F1_metric(pred, mask, num_cl = 1, class_names=[]):
    # have been checked just for binary segm
    
    TP = np.zeros((num_cl), dtype=int)
    TN = np.zeros((num_cl), dtype=int)
    FN = np.zeros((num_cl), dtype=int)
    FP = np.zeros((num_cl), dtype=int)
    
    #maskarr=np.asarray([mask])
    #pred=np.asarray([pred])
    maskarr =mask
    for cl in range(num_cl):
        gt = maskarr[:,:,:,cl]
        pr_cl = np.where(pred[:,:,:,cl] >= 0.5 * (num_cl == 1) + np.argmax(pred, axis=-1) * (num_cl != 1), 1, 0)
        TP[cl] += np.sum(pr_cl * gt)
        FP[cl] += np.sum(pr_cl * np.where(gt==0, 1, 0)*(np.sum(maskarr, axis=-1) * (num_cl != 1) \
                                                         + (num_cl == 1)))
        FN[cl] += np.sum(np.where(pr_cl==0, 1, 0) * gt )
        TN[cl] += np.sum(np.where(pr_cl==0, 1, 0) * np.where(gt==0, 1, 0)*(np.sum(maskarr, axis=-1) * (num_cl != 1) \
                                                         + (num_cl == 1)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN) 
    f1_cl = 2*((precision*recall)/(precision+recall))
    
    for i in range(num_cl):
        if len(class_names):
            print(classes[i], ': precision ', round(precision[i], 3), ' recall ', round(recall[i], 3), ' f1 ', round(f1_cl[i], 3))
        else:
            print(i, ': precision ', round(precision[i], 3), ' recall ', round(recall[i], 3), ' f1 ', round(f1_cl[i], 3))
        print(' ')
'''
def F1_metric(pred, mask, num_cl = 1, class_names=[]):
    # have been checked just for binary segm
    
    TP = np.zeros((num_cl), dtype=int)
    TN = np.zeros((num_cl), dtype=int)
    FN = np.zeros((num_cl), dtype=int)
    FP = np.zeros((num_cl), dtype=int)
    
    maskarr =mask
    for cl in range(num_cl):
        gt = maskarr[:,:,:,cl]
        pr_cl = pred[:,:,:,cl]
        #print(np.sum(pr_cl), np.sum(gt))
        TP[cl] += np.sum(pr_cl * gt)
        FP[cl] += np.sum(pr_cl * np.where(gt==0, 1, 0)*(np.sum(maskarr, axis=-1) * (num_cl != 1) \
                                                         + (num_cl == 1)))
        FN[cl] += np.sum(np.where(pr_cl==0, 1, 0) * gt )
        TN[cl] += np.sum(np.where(pr_cl==0, 1, 0) * np.where(gt==0, 1, 0)*(np.sum(maskarr, axis=-1) * (num_cl != 1) \
                                                         + (num_cl == 1)))
    #print(TP[2], FP[2])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN) 
    f1_cl = 2*((precision*recall)/(precision+recall))
    
    for i in range(num_cl):
        if len(class_names):
            print(class_names[i], ': precision ', round(precision[i], 3), ' recall ', round(recall[i], 3), ' f1 ', round(f1_cl[i], 3))
        else:
            print(i, ': precision ', round(precision[i], 3), ' recall ', round(recall[i], 3), ' f1 ', round(f1_cl[i], 3))
        print(' ')
     
    return f1_cl
def RMSE(prediction, mask):
    rmse = round(np.sqrt(np.sum(np.square(prediction - mask*np.where(prediction==0, 0, 1)))/np.sum(np.where(prediction==0, 0, 1))), 3)
    return rmse

def MAE(prediction, mask):
    mae = round(np.sum(np.abs(prediction - mask*np.where(prediction==0, 0, 1)))/np.sum(np.where(prediction==0, 0, 1)), 3)
    return mae

def MBE(prediction, mask):
    MBE = np.mean(mask - prediction)
    return MBE
    
def R2(prediction, mask):
    R2 = round(1 - np.sum((prediction-mask)**2)/np.sum((mask - np.mean(mask))**2), 3)
    return R2
