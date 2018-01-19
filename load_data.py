import numpy as np
from scipy.misc import imread
from scipy.misc import imresize as resize
import json
import os

def load_synth(index):
    ### type of index: list or 1d array
    ### index from 0~99999
    ### return imgs shape = (None, 240, 320, 3)
    ### return label shape = (None, 10)
    ### (R, x0, y0, x1, y1, L, x0, y0, x1, y1)
    ### Note that
    # the origin is the upper-left corner of the image
    # (x0, y0) is the position of the upper-left corner of the bounding box
    # (x1, y1) is the position of the lower-right corner of the bounding box
    # x refers to column and y refers to row in image array
    # x may be larger than 320 or less than 0
    # y may be larger than 240 or less than 0
    path1 = 'DeepQ-Synth-Hand-01/data'
    path2 = 'DeepQ-Synth-Hand-02/data'
    path = [path1, path2]
    sss = ['s000', 's001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009']
    img_namelist = np.load('img_namelist.npy')
    label_namelist = np.load('label_namelist.npy')
    N = len(index)
    x_train = np.zeros([N, 240, 320, 3])
    y_train = np.zeros([N, 10])
    for n,ind in enumerate(index):
        i = ind // 50000
        i_mod = ind % 50000
        j = i_mod // 10000
        k = i_mod % 10000
        # print(i,j,k)
        datapath = path[i]
        s_dir = sss[i*5+j]
        img = imread(os.path.join(datapath, s_dir, 'img', img_namelist[k]))
        x_train[n] = img
        f = open(os.path.join(datapath, s_dir, 'label', label_namelist[k]))
        bbox = json.load(f)['bbox']
        f.close()
        if 'R' in bbox.keys():
            y_train[n, 0] = 1
            y_train[n, 1:5] = bbox['R']
        if 'L' in bbox.keys():
            y_train[n, 5] = 1
            y_train[n, 6:10] = bbox['L']
    y_train = clean_synth_label(y_train)
    return x_train.astype(np.uint8), y_train


def clean_synth_label(y):
    ### shape of y = (None, 10)
    ### make sure that bbox is not out of range
    N = y.shape[0]
    for i in range(N):
        if y[i,1]<0: y[i,1]=0
        if y[i,2]<0: y[i,2]=0
        if y[i,3]>320: y[i,3]=320
        if y[i,4]>240: y[i,4]=240
        if y[i,6]<0: y[i,6]=0
        if y[i,7]<0: y[i,7]=0
        if y[i,8]>320: y[i,8]=320
        if y[i,9]>240: y[i,9]=240
    return y


def load_vivepaper():
    ### original image size is (460, 612, 3)
    path_air = os.path.join('DeepQ-Vivepaper', 'data', 'air')
    path_book = os.path.join('DeepQ-Vivepaper', 'data', 'book')
    air_imglist = np.sort(os.listdir(os.path.join(path_air, 'img')))
    air_labellist = np.sort(os.listdir(os.path.join(path_air, 'label')))
    book_imglist = np.sort(os.listdir(os.path.join(path_book, 'img')))
    book_labellist = np.sort(os.listdir(os.path.join(path_book, 'label')))
    n_air = air_imglist.shape[0]
    n_book = book_imglist.shape[0]

    x_air = np.zeros([n_air, 240, 320, 3])
    y_air = np.zeros([n_air, 10])
    x_book = np.zeros([n_book, 240, 320, 3])
    y_book = np.zeros([n_book, 10])
    for i, name in enumerate(air_imglist):
        img = imread(os.path.join(path_air, 'img', name))
        img = resize(img, (240, 320, 3))
        x_air[i] = img
    for i, name in enumerate(air_labellist):
        f = open(os.path.join(path_air, 'label', name))
        bbox = json.load(f)['bbox']
        f.close()
        if 'R' in bbox.keys():
            y_air[i, 0] = 1
            y_air[i, 1:5] = bbox['R']
        if 'L' in bbox.keys():
            y_air[i, 5] = 1
            y_air[i, 6:10] = bbox['L']
    for i, name in enumerate(book_imglist):
        img = imread(os.path.join(path_book, 'img', name))
        img = resize(img, (240, 320, 3))
        x_book[i] = img
    for i, name in enumerate(book_labellist):
        f = open(os.path.join(path_book, 'label', name))
        bbox = json.load(f)['bbox']
        f.close()
        if 'R' in bbox.keys():
            y_book[i, 0] = 1
            y_book[i, 1:5] = bbox['R']
        if 'L' in bbox.keys():
            y_book[i, 5] = 1
            y_book[i, 6:10] = bbox['L']
    y_air = resize_y(y_air)
    y_book = resize_y(y_book)
    return x_air.astype(np.uint8), y_air, x_book.astype(np.uint8), y_book


def resize_y(y):
    # resize label from scale (460, 612) to (240, 320)
    xratio = 320./612.
    yratio = 240./460.
    y[:, 1] *= xratio
    y[:, 3] *= xratio
    y[:, 6] *= xratio
    y[:, 8] *= xratio
    y[:, 2] *= yratio
    y[:, 4] *= yratio
    y[:, 7] *= yratio
    y[:, 9] *= yratio
    return y


def normalize_y(y):
    ### normalize label scale of (240, 320) to 0~1
    xratio = 1./320.
    yratio = 1./240.
    y[:, 1] *= xratio
    y[:, 3] *= xratio
    y[:, 6] *= xratio
    y[:, 8] *= xratio
    y[:, 2] *= yratio
    y[:, 4] *= yratio
    y[:, 7] *= yratio
    y[:, 9] *= yratio
    return y


if __name__ == '__main__':
    img, y = load_synth([0])
    print(img)

    print('load_vivepaperload_vivepaperload_vivepaper')
    x_air, y_air, x_book, y_book = load_vivepaper()
    print(x_air[0])

    #
    # import matplotlib.pyplot as plt
    # plt.imshow(img[0])
    # plt.show()

    # from make_ytrue_tensor import make_train_tensor
    # for i in range(10):
    #     x, y = load_synth(np.arange(i*10000, (i+1)*10000))
    #     np.save('./npies/' + 'x{}.npy'.format(i), x)
    #     np.save('./npies/' + 'y{}.npy'.format(i), make_train_tensor(y).astype(np.float32))

    # x, y = load_synth(np.arange(9*10000, (9+1)*10000))
    # np.save('./npies/' + 'x9_.npy', x[:9000])
    # np.save('./npies/' + 'x_val.npy', x[9000:])
    # np.save('./npies/' + 'y9_.npy', make_train_tensor(y[:9000]).astype(np.float32))
    # np.save('./npies/' + 'y_val.npy', make_train_tensor(y[9000:]).astype(np.float32))
    # x_air, y_air, x_book, y_book = load_vivepaper()
