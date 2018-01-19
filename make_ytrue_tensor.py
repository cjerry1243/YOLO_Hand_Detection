import load_data
import numpy as np
import matplotlib.pyplot as plt
### imgs shape = (None, 240, 320, 3)
### label shape = (None, 10)
### (R, x0, y0, x1, y1, L, x0, y0, x1, y1)


'''
make_train_tensor  fn
    input y (R, x0, y0, x1, y1, L, x0, y0, x1, y1)  
    return (6, 8, 7) vector
'''


def show_box(img, x0, y0, x1, y1):
    plt.imshow(img)
    plt.plot(range(x0, x1), np.repeat(y0, x1-x0), 'r')
    plt.plot(range(x0, x1), np.repeat(y1, x1-x0), 'r')
    plt.plot(np.repeat(x0, y1-y0), range(y0, y1), 'r')
    plt.plot(np.repeat(x1, y1-y0), range(y0, y1), 'r')
    plt.xlim((0, 320))
    plt.ylim((240, 0))
    plt.show()


def make_train_tensor(y):
    # img_R, img_L format (C, Xc, Yc, Width, Height, R, L)
    # input (R, x0, y0, x1, y1, L, x0, y0, x1, y1)

    num_samples = y.shape[0]


    matrix = np.zeros((num_samples, 6, 8, 7))

    for num in range(num_samples):

        img_R = np.zeros(7)
        img_L = np.zeros(7)

        if y[num, 0] == 1:
            img_R[0] = 1                              # C
            img_R[1] = (y[num, 1] + y[num, 3])/2      # Xc
            img_R[2] = (y[num, 2] + y[num, 4])/2      # Yc
            img_R[3] = (y[num, 3] - y[num, 1])/320    # Width
            img_R[4] = (y[num, 4] - y[num, 2])/240    # Height
            img_R[5] = 1                              # Right

        if y[num, 5] == 1:
            img_L[0] = 1                              # C
            img_L[1] = (y[num, 6] + y[num, 8])/2      # Xc
            img_L[2] = (y[num, 7] + y[num, 9])/2      # Yc
            img_L[3] = (y[num, 8] - y[num, 6])/320    # Width
            img_L[4] = (y[num, 9] - y[num, 7])/240    # Height
            img_L[6] = 1                              # Left

        R_x_index = int(img_R[1] // 40)
        R_y_index = int(img_R[2] // 40)

        L_x_index = int(img_L[1] // 40)
        L_y_index = int(img_L[2] // 40)

        '''
        parametrize the bounding box x and y coordinates 
        to be offsets of a particular grid cell location
        so they are also bounded between 0 and 1
        '''

        img_R[1] = img_R[1] / 40 - R_x_index
        img_R[2] = img_R[2] / 40 - R_y_index

        img_L[1] = img_L[1] / 40 - L_x_index
        img_L[2] = img_L[2] / 40 - L_y_index

        matrix[num, L_y_index, L_x_index, :] = img_L
        matrix[num, R_y_index, R_x_index, :] = img_R

    return matrix

if __name__ == '__main__':
    x, y = load_data.load_synth(list(range(1000)))
    a = make_train_tensor(y)









