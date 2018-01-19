from load_data import load_synth, load_vivepaper
from make_ytrue_tensor import make_train_tensor
from yolo_model import Vgg16_Yolo, Xception_Yolo, MobileNet_Yolo, ResNet_Yolo
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import time


λ_coord = 5.
λ_noobj = 0.5


def calc_loss(y_true, y_pred):
    losses = None
    for i in range(6):
        for j in range(8):
            if losses is None:
                losses = grid_loss(y_pred[:, i, j], y_true[:, i, j])
            else:
                losses += grid_loss(y_pred[:, i, j], y_true[:, i, j])

    return K.mean(losses, axis=0)


def grid_loss(B_pred, B_true):
    #         0     1  2  3  4  5     6  7  8  9  10      11
    # B_pred:(Conf, x, y, w, h, Conf, x, y, w, h, R_prob, L_prob)
    # B_true:(Conf, x, y, w, h, R_p,  L_p)
    # return loss of Conf, xy, wh

    c1 = B_pred[..., 0]
    c2 = B_pred[..., 5]

    B_response = compare_iou(B_pred, B_true)  # (?, 5)
    x_y = B_true[..., 1:3]
    w_h = B_true[..., 3:5]
    c = B_true[..., 0]
    R_L_prob = B_true[..., 5:7]

    x_y_hat = B_response[..., 1:3]
    w_h_hat = B_response[..., 3:5]
    c_hat = B_response[..., 0]
    R_L_prob_hat = B_pred[..., 10:12]

    x_y_error = λ_coord * K.sum(K.square(x_y - x_y_hat), axis=-1)
    w_h_error = λ_coord * K.sum(K.square(K.sqrt(w_h) - K.sqrt(w_h_hat)), axis=-1)
    conf_error = K.square(c - c_hat)
    prob_error = K.sum(K.square(R_L_prob - R_L_prob_hat), axis=-1)

    object_error = x_y_error + w_h_error + conf_error + prob_error
    no_object_error = λ_noobj * (K.square(c - c1) + K.square(c - c2))

    iou_loss = B_true[..., 0] * object_error + (1. - B_true[..., 0]) * no_object_error
    # print(iou_loss.shape)

    return iou_loss


def compare_iou(B_pred, B_true):

    B1 = B_pred[..., 1:5]
    B2 = B_pred[..., 6:10]
    BT = B_true[..., 1:5]

    iou1 = IOU(B1, BT)
    iou2 = IOU(B2, BT)

    iou = K.concatenate([iou1, iou2], axis=-1)
    switch = K.one_hot(K.argmax(iou), num_classes=2)
    # chosen_B_pred = switch[:, 0] * B_pred[..., :5] + switch[:, 1] * B_pred[..., 5:10]   # ERROR
    chosen_B_pred = switch[:, 0:1] * B_pred[..., :5] + switch[:, 1:2] * B_pred[..., 5:10]
    return chosen_B_pred # (?, 5)


def IOU(B_pred_4, B_true_4):
    # B: (x, y, w, h)
    assert B_pred_4.shape[1] == 4, 'B_pred_4 must only have (x, y, w, h)'

    B_pred_min = B_pred_4[..., :2] - B_pred_4[..., 2:] / 2
    B_pred_max = B_pred_4[..., :2] + B_pred_4[..., 2:] / 2
    B_true_min = B_true_4[..., :2] - B_true_4[..., 2:] / 2
    B_true_max = B_true_4[..., :2] + B_true_4[..., 2:] / 2
    intersect_min = K.maximum(B_pred_min, B_true_min)
    intersect_max = K.minimum(B_pred_max, B_true_max)
    intersect_wh = K.maximum(intersect_max - intersect_min, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_areas = B_pred_4[..., -2] * B_pred_4[..., -1]
    true_areas = B_true_4[..., -2] * B_true_4[..., -1]
    iou = intersect_areas / (pred_areas + true_areas - intersect_areas)

    return K.reshape(iou, shape=[-1, 1])


def batch_generator(batchsize, minibatch_size=10000, batch_max=100000):
    # shape of batch_x  = (batchsize, 240, 320, 3)
    # shape of batch_y = (batchsize, 6, 8, 7)

    def load_npy(batch_count, minibatch_size):
        # return shuffled dataset
        shuffle = np.random.permutation(minibatch_size)
        file_index = batch_count // minibatch_size
        data_x = np.load('./npies/x%d.npy' % file_index)
        data_y = np.load('./npies/y%d.npy' % file_index)
        # print('load%d' % file_index)
        return data_x[shuffle], data_y[shuffle]

    start = 0
    batch_count = 0
    minibatch_x, minibatch_y = load_npy(batch_count, minibatch_size)

    while True:

        if batch_count >= batch_max:
            start = 0
            batch_count = 0
            minibatch_x, minibatch_y = load_npy(batch_count, minibatch_size)

        if start + batchsize > minibatch_size:
            start = 0
            minibatch_x, minibatch_y = load_npy(batch_count, minibatch_size)

        end = start + batchsize

        batch_x = minibatch_x[start:end]
        batch_y = minibatch_y[start:end]

        batch_count += batchsize
        start = end

        yield batch_x / 255., batch_y

    # shuffle = np.random.permutation(10000)
    # end = 0
    # while True:
    #     start = end if end < 10000 else 0
    #     end = min(start + batchsize, 10000)
    #
    #     index = shuffle[start:end]
    #     end = end%100000
    #
    #     batch_x, batch_y = load_synth(index)
    #     batch_y = make_train_tensor(batch_y)
    #     yield batch_x, batch_y


def batch_vive_generator(batchsize, minibatch_size=10000, batch_max=100000):
    # shape of batch_x  = (batchsize, 240, 320, 3)
    # shape of batch_y = (batchsize, 6, 8, 7)

    def load_npy(batch_count, minibatch_size):
        # return shuffled dataset
        shuffle = np.random.permutation(minibatch_size)
        file_index = batch_count // minibatch_size
        data_x = np.load('./npies/x%d.npy' % file_index)
        data_y = np.load('./npies/y%d.npy' % file_index)
        # print('load%d' % file_index)
        return data_x[shuffle], data_y[shuffle]

    start = 0
    batch_count = 0
    minibatch_x, minibatch_y = load_npy(batch_count, minibatch_size)

    while True:

        if batch_count >= batch_max:
            start = 0
            batch_count = 0
            minibatch_x, minibatch_y = load_npy(batch_count, minibatch_size)

        if start + batchsize > minibatch_size:
            start = 0
            minibatch_x, minibatch_y = load_npy(batch_count, minibatch_size)

        end = start + batchsize

        batch_x = minibatch_x[start:end]
        batch_y = minibatch_y[start:end]

        batch_count += batchsize
        start = end

        yield batch_x / 255., batch_y


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.filename = './weights/fix_resnet_weights/%s_loss_log.txt' % time.strftime('%Y%m%d-%H%M')

    def on_epoch_begin(self, epoch, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        # self.losses.append(logs.get('loss'))
        t_str = time.strftime('%Y%m%d-%H%M%S')
        print(t_str, logs.get('loss'), file=open(self.filename, 'a'))

    def on_epoch_end(self, epoch, logs={}):
        # t_str = time.strftime('%Y%m%d-%H%M%S')
        # print(t_str, self.losses, file=open(self.filename, 'a'))
        pass



batchsize = 300
maxepoch = 200
batch_max = 90000

if __name__ == '__main__':

    # yolo = Vgg16_Yolo()
    # yolo = Xception_Yolo()
    yolo = ResNet_Yolo(train_resnet=False)
    # yolo = MobileNet_Yolo()
    # yolo = build_model(model_name='Mobile')
    yolo.compile(optimizer='adam', loss=calc_loss)
    checkpointer = ModelCheckpoint(filepath='./weights/fix_resnet_weights/yolo_weights_{epoch:02d}_{loss:.2f}.hdf5', verbose=0, monitor='loss',
                                   save_best_only=True, save_weights_only=True)
    losslogger = LossHistory()

    batch_gen = batch_generator(batchsize, minibatch_size=10000, batch_max=batch_max)

    # for i in range(400):
    #     x, y = next(batch_gen)

    yolo.fit_generator(batch_gen, epochs=maxepoch, steps_per_epoch=batch_max/batchsize, callbacks=[checkpointer, losslogger])
