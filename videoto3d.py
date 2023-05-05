# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:05:14 2022

@author: TUF
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import random

def normalize_image( image, mean, std ):
    for channel in range(3):
        image[:,:,channel] = (image[:,:,channel] - mean[channel])/ std[channel]
    return image

def crop_center_square(frame):
     #print( frame.shape )
     y, x = frame.shape[0:2]
     min_dim = min(y, x)
     #print('x:', x,'\ny:',y)
     start_x = (x // 2) - (min_dim // 2)
     start_y = (y // 2) - (min_dim // 2)
     #print( start_y, start_y + min_dim, start_x, start_x + min_dim )
     return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        

    def video3d(self, filename, n, color=True, skip=True):
        print("filename: ",filename)
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = []
        if skip:
            for x in range( self.depth):
                frames.append(x * nframe / self.depth) 
            while len(frames)!=self.depth:
                frames.append([ (self.depth-1) * nframe / self.depth + 1])
        else:
            frames = [x for x in range(self.depth)]
        framearray = []
        #if n == 3 or n == 0:
         #   rotate_angle = random.uniform(-30.0, 30.0)

        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            
            if not ret:
                break
            
            #if n > 0:
             #  gen = ImageDataGenerator()
              # if n == 3:
               #     frame = gen.apply_transform( frame, {'theta':rotate_angle} )
               #if n == 4:
               #     frame = gen.apply_transform( frame, {'ty': 10} )
            
            frame = crop_center_square(frame)
            try:
                if color :
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                frame = cv2.resize(frame, (self.height, self.width))
                #frame = normalize_image(np.array(frame) / 255.0,
                #                       mean = [0.485, 0.456, 0.406],
                #                      std = [0.229, 0.224, 0.225])
                #frame = frame.astype('float32')
                framearray.append(frame)
            except Exception as e:
                 print(str(e))

        cap.release()
        return np.array(framearray)
