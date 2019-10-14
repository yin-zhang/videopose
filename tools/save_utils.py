import numpy as np
import sys
import cv2
import os
import struct

CV_8U   = 0
CV_8S   = 1
CV_16U  = 2
CV_16S  = 3
CV_32S  = 4
CV_32F  = 5
CV_64F  = 6

def writeByte(f, v):
    f.write(struct.pack('b', v))

def writeUnsignedByte(f, v):
    f.write(struct.pack('B', v))

def writeShort(f, v):
    f.write(struct.pack('h', v))

def writeUnsignedShort(f, v):
    f.write(struct.pack('H', v))

def writeInt(f, v):
    f.write(struct.pack('i', v))

def writeUnsignedInt(f, v):
    f.write(struct.pack('I', v))

def writeFloat(f, v):
    f.write(struct.pack('f', v))

def writeDouble(f, v):
    f.write(struct.pack('d', v))

def quantize(bmin, bmax, qmin, qmax):
    # extend the [bmin, bmax] interval to include 0.
    bmin = min(bmin, 0.0)
    bmax = max(bmax, 0.0)
    if bmin == bmax:
        scale = 1.0
        bias = 0.0
    else:
        scale = (qmax - qmin) / (bmax - bmin)
        bias = min(qmax, max(qmin, np.round(qmin - bmin * scale)))
    return scale, bias
    
def saveFloatMat(f, t, A):
    B = A.reshape(A.shape[0],-1)
    bmax = max(B.reshape(-1))
    bmin = min(B.reshape(-1))
    
    if t == CV_8U:
        # CV_8U
        scale, bias = quantize(bmin, bmax, 0.0, 255.0)
        B = np.uint8(np.round(B * scale + bias))
    elif t == CV_16U:
        # CV_16U
        scale, bias = quantize(bmin, bmax, 0.0, 65535.0)
        B = np.uint16(np.round(B * scale + bias))
    elif t == CV_32S:
        scale = 1.0
        bias = 0.0
        B = np.int32(B)
    elif t == CV_32F:
        scale = 1.0
        bias = 0.0
        B = np.float32(B)
    else:
        # CV_32F
        scale = 1.0
        bias = 0.0
        
    # rows
    writeInt(f, B.shape[0])
    # cols
    writeInt(f, B.shape[1])
    # channels
    writeInt(f, 1)
    # type
    writeInt(f, t)
    # scale, bias
    writeDouble(f, scale)
    writeDouble(f, bias)
    # write elements
    B = B.reshape(-1)
    for i in range(B.shape[0]):
        if t == CV_8U:
            writeUnsignedByte(f, B[i])
        elif t == CV_16U:
            writeUnsignedShort(f, B[i])
        elif t == CV_32S:
            writeInt(f, B[i])
        elif t == CV_32F:
            writeFloat(f, B[i])
        else:
            assert(False)

def saveFloatCvTensorList(f, mat_list, type_list):
    N = len(mat_list)
    writeInt(f, N)
    for i in range(N):
        saveFloatMat(f, type_list[i], mat_list[i])

def save_prediction(out_file, prediction, input_keypoints, cam_w, cam_h):

    ext = os.path.splitext(out_file)[-1].lower()

    if ext == '.dat':
        mat_list = []
        type_list = []

        mat_list.append(prediction)
        type_list.append(CV_32F)
    
        mat_list.append(input_keypoints)
        type_list.append(CV_32F)

        meta = np.int32([cam_w, cam_h])
        mat_list.append(meta)
        type_list.append(CV_32S)
    
        f = open(out_file, 'wb')
        saveFloatCvTensorList(f, mat_list, type_list)
        f.close()
    else:
        np.savez_compressed(out_file, prediction=prediction, input_keypoints=input_keypoints, cam_w=cam_w, cam_h=cam_h)
        
