import numpy as np
import os
from PIL import Image, ImageOps
from pathlib import Path
from random import shuffle, randint
import sys
import scipy.misc
from utils import *
class covers:
    def __init__(self, folder, batch_size, cat_dim):
        if "all covers" in folder:
            image_list = '/image_list_one_million.txt'
            print("One Million Cover Art")
        elif "spotify" in folder:
            image_list = '/image_list_spotify.txt'
            print("Spotify dataset")
        else:
            image_list = 'none'
        is_file = os.path.isfile('.' + image_list)
        self.list_file = Path(os.getcwd() + image_list)
        self.directory = folder
        self.batch_size = batch_size
        self.list = []
        self.num_images = 0
        self.cat_dim = cat_dim
        if is_file:
            print('Loading list of images from existing text file.')
            with open(str(self.list_file)) as f:
                self.list = f.read().splitlines()
            self.num_images = len(self.list)
            print (self.num_images)
        else:
            self.generate_list()
        self.generate_catagories()

    def generate_catagories(self):
        self.cat = np.zeros([self.num_images])
        num = 0
        rand = False
        if 'all covers' in self.directory:
            rand = True
        for l in self.list:
            if rand:
                #print('random genre tags used')
                self.cat[num] = randint(0,4)
            elif 'jazz' in l:
                self.cat[num] = 0
            elif 'dance' in l:
                self.cat[num] = 1
            elif 'rock' in l:
                self.cat[num] = 2
            elif 'rap' in l:
                self.cat[num] = 3
            elif 'metal' in l:
                self.cat[num] = 4
            num += 1

    def generate_list(self):
        txt = open('image_list.txt', 'w')
        for path, subdirs, files in os.walk(self.directory):
            for filename in files:
                f = (path + '/' + filename)
                out = (str(f))
                # Check if the file is a valid image
                try:
                    img = Image.open(f)
                    self.list.append(out)
                    txt.write(out + '\n')
                except Exception as e:
                    print (e)
        txt.close()
        self.num_images = len(self.list)

    def get_image(self, start_idx=None):
        for i in range(len(self.list)):
            if start_idx is None or start_idx <= i:
                img = Image.open(self.list[i])
                catagory = self.cat[i]
                try:
                    img2 = ImageOps.fit(img, [64, 64], Image.ANTIALIAS)
                    rgbimg = img2.convert('RGB')
                    rgbimg = np.array(rgbimg, dtype=np.float32)/127.5 - 1
                except Exception as e:
                    continue
                yield i, rgbimg, catagory

    #def get_image(self, start_idx=None):
    #    for i in range(len(self.list)):
    #        if start_idx is None or start_idx <= i:
    #            image = imread(self.list[i])
    #            rgbimg = transform(image, 64, 64, 64 , 64, True)
    #            yield i, rgbimg

    # get next batch of photos
    def batched_images(self, start_idx=None):
        batch, next_idx, data_counter = None, None, 0
        for idx, image, img_cat in self.get_image(start_idx):
            if batch is None:
                batch = np.empty((self.batch_size, 64, 64, 3))
                batch_cat = np.empty((self.batch_size), dtype=int)
                next_idx = 0
                data_counter = 0
            if image is not None:
                batch[data_counter] = image
                batch_cat[data_counter] = img_cat
                data_counter += 1
            else:
                print("exception")
            next_idx += 1
            if data_counter == self.batch_size:
                yield next_idx + 1, batch, batch_cat
                batch = None
