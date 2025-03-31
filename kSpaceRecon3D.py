# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:18:02 2019

@author: Tom O'Reilly
"""

import numpy as np 
import matplotlib.pyplot as plt
import interactive3Dplot as plt3D
import keaDataProcessing as keaProc
import imageProcessing as imProc
import distortionCorrection as distCorr

dataFolder = r'20221214 - Tom ACR/TSEV3_1_highGrad'
experimentNumber = 4

scanParams = keaProc.readPar(r'%s/%i/acqu.par'%(dataFolder,experimentNumber))

'''Read k-space data'''
kSpace, scanParams = keaProc.readKSpace(r'%s/%i/data.3d'%(dataFolder,experimentNumber), scanParams = scanParams, correctOversampling = True)

'''Noise correction'''
kSpace = imProc.noiseCorrection(kSpace, scanParams, dataFolder)

'''apply filter to k-space data'''
kSpace          = imProc.sineBellSquaredFilter(kSpace, filterStrength = 0.0) # strength = 0 no filter, 1 = max

'''Distortion correction'''

#B0 correction
reconImage = distCorr.b0Correction(kSpace, scanParams, dataFolder, shOrder = 8)

#gradient correction
reconImage = distCorr.gradUnwarp(reconImage, scanParams, dataFolder)

'''post processing''' #data should revert back to k-space for this
# kSpace = np.fft.fftshift(np.fft.ifftn((np.fft.fftshift(reconImage))))

'''Zero fill data'''
#kSpace       = imProc.zeroFill(kSpace, (180,180,16))

''' Shift image'''
# shiftDistance = 0.025 #shift distance in meteres
# shiftAxis = 2 # FE = 0, phase 1 = 1, phase 2 = 2
# kSpace = imProc.shiftImage(kSpace, scanParams, shiftDistance,shiftAxis)

# reconImage       = np.fft.fftshift(np.fft.fftn((np.fft.fftshift(kSpace))))

# reconImage = np.flip(reconImage, axis = 0)

fig, ax = plt.subplots()
fig3D = plt3D.interactivePlot(fig, ax, np.abs(reconImage), plotAxis = 2, fov = scanParams["FOV"], axisLabels = scanParams["axisLabels"])


nColumns    = 12
nRows       = int(np.floor(np.size(reconImage,-1)/nColumns))
nRows = 8

cutSlice = np.size(reconImage, -1) - nColumns*nRows

vmin = np.min(np.abs(reconImage[:,:,cutSlice:-cutSlice]))
vmax = np.max(np.abs(reconImage[:,:,cutSlice:-cutSlice]))

fig = plt.figure(figsize=(nColumns,nRows)) # Notice the equal aspect ratio
ax = [fig.add_subplot(nRows,nColumns,i+1) for i in range(nRows*nColumns)] 
for idx in range(nColumns*nRows):
    ax[idx].imshow(np.abs(reconImage[:,:,idx+int(cutSlice/2)]), cmap  = 'gray', vmin = vmin, vmax = vmax)
    ax[idx].axis('off')
    ax[idx].set_aspect('equal')
fig.subplots_adjust(wspace=0, hspace=0)        
        