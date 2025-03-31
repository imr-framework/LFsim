# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:49:50 2019

@author: Tom O'Reilly

Version: 2.0
"""

import numpy as np
import struct

def readPar(filename):
    parFile = open(filename, 'r') 
    
    parameters = {}
    
    for line in parFile:
        parData = line.split()
        if len(parData) == 3:
            parameters[parData[0]] = parData[2]
    '''Determine orientation of image'''
    if "plane" in parameters:
        if parameters["plane"][1] == "x":
            if parameters["plane"][2] == "y":
                parameters["axisLabels"] = ["Foot/Head", "Anterior/Posterior", "Left/Right"]
            else:
                parameters["axisLabels"] = ["Foot/Head", "Left/Right", "Anterior/Posterior"]
        elif parameters["plane"][1] == "y":
            if parameters["plane"][2] == "z":
                parameters["axisLabels"] = ["Anterior/Posterior", "Left/Right", "Foot/Head"]
            else:
                parameters["axisLabels"] = ["Anterior/Posterior", "Foot/Head", "Left/Right"]
        elif parameters["plane"][1] == "z":
            if parameters["plane"][2] == "y":
                parameters["axisLabels"] = ["Left/Right", "Anterior/Posterior", "Foot/Head"]
            else:
                parameters["axisLabels"] = ["Left/Right", "Foot/Head", "Anterior/Posterior"]
        else:
            print("Invalid plane orientation")
            parameters["axisLabels"] = ["unknown", "unknown", "unknown"]
        try:
            parameters["FOV"] = [float(parameters['FOVread']), float(parameters['FOVphase1']), float(parameters['FOVphase2'])]
        except:
            print("Error reading field of view, could not find all/some of the parameters: FOVread, FOVPhase1, FOVPhase2")
            print("Defaulting to 256, 256, 256 mm field of view")
            parameters["FOV"] = [256, 256, 256]
            
    parameters["parFile"] = filename
        
    return parameters

def readRealMatrix(filename):
    header = []
    f = open(filename,"rb")
    blocksize = 4
    for idx in range(8):
        header.append(f.read(blocksize))
    dim1 = struct.unpack('<i',header[4])[0]
    dim2 = struct.unpack('<i',header[5])[0]
    dim3 = struct.unpack('<i',header[6])[0]
    dim4 = struct.unpack('<i',header[7])[0]
    dataSize = dim1*dim2*dim3*dim4
    rawData = f.read()
    f.close()

    rawData = np.array(struct.unpack(str(dataSize)+'f',rawData))
    if dim4 != 1:   #4 dimensional array
        return np.reshape(rawData.T, (dim1, dim2, dim3, dim4), order = 'F')
    elif dim3 != 1:
        return np.reshape(rawData.T, (dim1, dim2, dim3), order = 'F')
    elif dim2 != 1:
        return np.reshape(rawData.T, (dim1, dim2), order = 'F')    
    elif dim1 != 1:
        return np.reshape(rawData.T, (dim1), order = 'F')
    
def readKSpace(filename, scanParams = {}, correctOversampling = True, forceOversamplingCorrection = False):
    header = []
    f = open(filename,"rb")
    blocksize = 4
    for idx in range(8):
        header.append(f.read(blocksize))
    dim1 = struct.unpack('<i',header[4])[0]
    dim2 = struct.unpack('<i',header[5])[0]
    dim3 = struct.unpack('<i',header[6])[0]
    dim4 = struct.unpack('<i',header[7])[0]
    dataSize = 2*dim1*dim2*dim3*dim4
    rawData = f.read()
    f.close()

    rawData = np.array(struct.unpack(str(dataSize)+'f',rawData))
    rawData = rawData[::2] + 1j*rawData[1::2]
    
    if dim4 != 1:   #4 dimensional array
        reshapedData = np.reshape(rawData.T, (dim1, dim2, dim3, dim4), order = 'F')
    elif dim3 != 1:
        reshapedData = np.reshape(rawData.T, (dim1, dim2, dim3), order = 'F')
    elif dim2 != 1:
        reshapedData = np.reshape(rawData.T, (dim1, dim2), order = 'F')    
    elif dim1 != 1:
        reshapedData = np.reshape(rawData.T, (dim1), order = 'F')
    else:
        print("Invalid data shape, data may be corrupted or may contain more than 4 dimensions")
    
    if((("overSampling" in scanParams and scanParams["overSampling"] == "\"yes\"") or forceOversamplingCorrection) and correctOversampling):
        import scipy.signal as sig
        scanParams["reconBW"] = float(scanParams["bandwidth"])/2
        return sig.decimate(reshapedData, 2 , ftype = 'fir', axis = 0), scanParams
    else:
        try:
            scanParams["reconBW"] = float(scanParams["bandwidth"])
            return reshapedData, scanParams
        except:
            return reshapedData

def autoPhaseMaximise(data, resolution = 1):
    phaseAngles = np.arange(0,360,resolution)
    angledData = np.outer(data, np.exp(-1j*phaseAngles*np.pi/180))
    angledData = np.sum(np.real(angledData), axis = 0)
    zeroPhaseAngle = phaseAngles[np.argmax(angledData)]
    print('Zeroth order phase correction: ' + str(zeroPhaseAngle) + ' degrees')
    return(np.multiply(np.exp(-1j*zeroPhaseAngle*np.pi/180), data), zeroPhaseAngle)


    