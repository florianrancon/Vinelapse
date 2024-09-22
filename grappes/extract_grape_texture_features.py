import pyfeats
import os
from os import listdir
import numpy as np
import pandas as pd
import cv2
import random
import phate
import scprep

liste_patch_size=[64,48,32]
write_patch=False
liste_folders=['fixcam10_AME','fixcam5_AME','fixcam7_AME']
folder_dir = "C://Users//Utilisateur//Documents//IMS//VINELAPSE//GRAPPES//" #Dossier contenant les masques de grappe
patch_dir= "C://Users//Utilisateur//Documents//IMS//VINELAPSE//GRAPPES//fixcam10_patches" #utilisé pour sauvegardé les patchs si besoin
out_dir= "C://Users//Utilisateur//Documents//IMS//VINELAPSE//GRAPPES//features"
p_trans_threshold=0.5 #Seuil sur la proportion de pixels non transparents (= de pixels de grappe) dans un patch pour déterminer s'il est traité
patch_superpos_factor=1.25 #facteur de superposition entre patchs. Si égal à 2 => 50% de superposition, si égal à 1, aucune superposition
def extract_patch(image,patch_center,patch_size):
    # calc patch position and extract the patch
    patch_x = int(patch_center[0] - patch_size / 2.)
    patch_y = int(patch_center[1] - patch_size / 2.)
    patch_image = image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]
    return patch_image

def load_image( infilename ) :
    img = cv2.imread(infilename, cv2.IMREAD_UNCHANGED)  
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    alpha_channel = img[:,:,3]
    hue_channel=hsv[:,:,0]
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    return img_grey,img,alpha_channel,hue_channel #contient une structure numpy


data = []
for folder in liste_folders:
    print('Folder: '+str(folder))
    for patch_size in liste_patch_size:
        print('patch size: '+str(patch_size))
        z=0
        i_global=0
        for images in os.listdir(folder_dir+folder):
            i_patch=0
            # check if the image ends with png
            if (images.endswith(".png")):
                z=z+1
                print(images)
                img,img_full,mask,hue=load_image(folder_dir+folder+"//"+images)
                height, width = img.shape[:2]
                idx =  cv2.findNonZero(mask)
                #Créer une grille de points sur l'image entière et sélectionner les patchs avec moins de la moitié des pixels en dehors du masque
                minX, maxX, minY, maxY = patch_size, height-patch_size-1, patch_size, width-patch_size-1
                x=list(range(minX, maxX, round(patch_size/patch_superpos_factor)))
                y=list(range(minY, maxY, round(patch_size/patch_superpos_factor)))
                X, Y = np.meshgrid(x, y)
                X=X.flatten()
                Y=Y.flatten()
                #print(X)
                #print(Y)
                for center_x, center_y in zip(X, Y):
                    patch_center=(center_x,center_y)
                    #Get the patch
                    patch_image=extract_patch(img,patch_center,patch_size)
                    patch_mask=extract_patch(mask,patch_center,patch_size)
                    height, width = patch_mask.shape[:2]
                    #check the proportion of transparent pixel
                    p_trans=cv2.countNonZero(patch_mask)/(patch_size*patch_size)
                    if (p_trans>p_trans_threshold): #Compute features on the patch if more than half of pixels are from a grape
                        i_patch=i_patch+1
                        i_global=i_global+1
                        out_patch = cv2.bitwise_and(patch_image,patch_image,mask = patch_mask)
                        if (write_patch):
                            cv2.imwrite(patch_dir+"//"+str(patch_size)+"_"+str(z)+"_"+str(i_patch)+".png", out_patch)
                        features, labels = pyfeats.fos(patch_image,patch_mask)
                        for f, l in zip(features, labels):
                            data.append([folder,patch_size,images,z,i_global,i_patch,l,f])
                        features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(patch_image, ignore_zeros=True)
                        for f, l in zip(features_mean, labels_mean):
                            data.append([folder,patch_size,images,z,i_global,i_patch,l,f])
                        features, labels = pyfeats.lbp_features(patch_image, patch_mask, P=[8,16,24], R=[1,2,3])
                        for f, l in zip(features, labels):
                            data.append([folder,patch_size,images,z,i_global,i_patch,l,f])
                        features, labels = pyfeats.glds_features(patch_image, patch_mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
                        for f, l in zip(features, labels):
                            data.append([folder,patch_size,images,z,i_global,i_patch,l,f])
                        features, labels = pyfeats.sfm_features(patch_image, patch_mask, Lr=4, Lc=4)
                        for f, l in zip(features, labels):
                            data.append([folder,patch_size,images,z,i_global,i_patch,l,f])
                        # features, Ht, labels = pyfeats.correlogram(patch_image, patch_mask, bins_digitize=5, bins_hist=5, flatten=True)
                        # for f, l in zip(features, labels):
                        #     data.append([z,i_global,i_patch,l,f])
                        features, labels = pyfeats.dwt_features(patch_image, patch_mask, wavelet='bior3.3', levels=3)
                        for f, l in zip(features, labels):
                            data.append([folder,patch_size,images,z,i_global,i_patch,l,f])
                        features, labels = pyfeats.swt_features(patch_image, patch_mask, wavelet='bior3.3', levels=3)
                        for f, l in zip(features, labels):
                            data.append([folder,patch_size,images,z,i_global,i_patch,l,f])
                        #compute mean of several channels, filtering dark bright/parts to avoid hue shifts
                        mask_light=cv2.inRange(patch_image, 30, 220)
                        #cv2.imwrite(patch_dir+"//"+str(z)+"_"+str(i_patch)+"_light.png", mask_light & patch_mask)
                        #cv2.imwrite(patch_dir+"//"+str(z)+"_"+str(i_patch)+"_mask.png", patch_mask)
                        #Hue
                        patch_hue=extract_patch(hue,patch_center,patch_size)
                        hue_mean=cv2.mean(patch_hue,mask=mask_light & patch_mask)
                        data.append([folder,patch_size,images,z,i_global,i_patch,"hue_mean",hue_mean[0]])
                        #Red
                        patch_R=extract_patch(img_full[:,:,2],patch_center,patch_size)
                        R_mean=cv2.mean(patch_R,mask=mask_light & patch_mask)
                        data.append([folder,patch_size,images,z,i_global,i_patch,"R_mean",R_mean[0]])
                        #Green
                        patch_G=extract_patch(img_full[:,:,1],patch_center,patch_size)
                        G_mean=cv2.mean(patch_G,mask=mask_light & patch_mask)
                        data.append([folder,patch_size,images,z,i_global,i_patch,"G_mean",G_mean[0]])
                        #Blue
                        patch_B=extract_patch(img_full[:,:,0],patch_center,patch_size)
                        B_mean=cv2.mean(patch_B,mask=mask_light & patch_mask)
                        data.append([folder,patch_size,images,z,i_global,i_patch,"B_mean",B_mean[0]])
                print("Extracted "+ str(i_patch)+ " patches")
        # phate_op = phate.PHATE()
        # df_wide=df_wide.dropna(axis='columns')
        # data_phate = phate_op.fit_transform(df_wide)
        # scprep.plot.scatter2d(data_phate, c=list(df_wide.index.get_level_values(0)), figsize=(12,8), cmap="Spectral",ticks=False, label_prefix="PHATE")
        # np.savetxt('C:/Users/flori/OneDrive/Desktop/vinelapse_2024_AE_grappes/out/feat_grey_PHAT'+str(patch_size)+'.csv', data_phate, delimiter=",")
df = pd.DataFrame(data)
df.columns = ["folder","patch_size","image","i_image","i_global", "i_patch", "label","feature"]
print(df.head(50))
df.to_csv(out_dir+'//feat_vinelapse.csv', index=False)
#df_wide=df.pivot(index=['image','i_patch'], columns='label', values='feature')
#df_wide.to_csv(out_dir+'//feat_grey_wide'+str(patch_size)+'_'+'.csv', index=False)
#print(df_wide.info())
#print(df_wide.head(50))