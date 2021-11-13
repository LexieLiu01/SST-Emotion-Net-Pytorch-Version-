import argparse
import configparser
import os
# from torch.optim import SGD, Adam, ASGD, RMSprop
# import numpy as np
# import torch
from model.model_pt import SST

import numpy as np 
# import time
# import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F
from configparser import ConfigParser
from collections import OrderedDict


train_specInput_root_path = None
train_tempInput_root_path = None
train_label_root_path = None

test_specInput_root_path = None
test_tempInput_root_path = None
test_label_root_path = None

result_path = None
model_save_path = None

input_width = None
specInput_length = None
temInput_length = None

depth_spec = None
depth_tem = None
gr_spec = None
gr_tem = None
nb_dense_block = None
nb_class = None

nbEpoch = None
batch_size = None
lr = None



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def read_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path)

    global train_specInput_root_path, train_tempInput_root_path, train_label_root_path, test_specInput_root_path, test_tempInput_root_path, test_label_root_path
    train_specInput_root_path = conf['path']['train_specInput_root_path']
    train_tempInput_root_path = conf['path']['train_tempInput_root_path']
    train_label_root_path = conf['path']['train_label_root_path']
    test_specInput_root_path = conf['path']['test_specInput_root_path']
    test_tempInput_root_path = conf['path']['test_tempInput_root_path']
    test_label_root_path = conf['path']['test_label_root_path']

    global result_path, model_save_path
    result_path = conf['path']['result_path']
    model_save_path = conf['path']['model_save_path']

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    global input_width, specInput_length, temInput_length
    input_width = int(conf['data']['input_width'])
    specInput_length = int(conf['data']['specInput_length'])
    temInput_length = int(conf['data']['temInput_length'])

    global depth_spec, depth_tem, gr_spec, gr_tem, nb_dense_block, nb_class
    depth_spec = int(conf['model']['depth_spec'])
    depth_tem = int(conf['model']['depth_tem'])
    gr_spec = int(conf['model']['gr_spec'])
    gr_tem = int(conf['model']['gr_tem'])
    nb_dense_block = int(conf['model']['nb_dense_block'])
    nb_class = int(conf['model']['nb_class'])

    global nbEpoch, batch_size, lr
    nbEpoch = int(conf['training']['nbEpoch'])
    batch_size = int(conf['training']['batch_size'])
    lr = float(conf['training']['lr'])

def to_tensor(x, fv=True):
    if fv:
        # return torch.FloatTensor(x).cuda(1)
        return torch.FloatTensor(x)
    else:
        # return torch.LongTensor(x).cuda(1)
        return torch.LongTensor(x)

def train(train_specInput, train_tempInput, train_label, sst, loss_func, opt):

    for epoch in range(nbEpoch):
        output = sst(train_specInput, train_tempInput)

        opt.zero_grad()
        loss = loss_func(output, train_label)
        loss.backward()
        opt.step()
        if (epoch+1) % 10 == 0:
            print("epoch : ", epoch, loss)

def evaluate(test_specInput, test_tempInput, test_label, sst, loss_func):
    sst.eval()
    output = sst(test_specInput, test_tempInput)
    eval_loss = loss_func(output, test_label)
    print("eval loss : ", eval_loss)
    
    output = output.detach().cpu().data.numpy()
    test_label = test_label.cpu().data.numpy()

    preds = np.argmax(output, axis=1).tolist()
    
    acc = 0
    for i in range(len(preds)):
        pred = preds[i]
        label = test_label[i]
        if pred == label:
            acc += 1
    print("total test acc : ", acc/len(preds))

    return acc/len(preds)

def run():
    all_result_file = open(os.path.join(result_path, 'all_result.txt'), "w")
    all_result_file.close()

    # sst = SST(input_width=input_width, specInput_length=specInput_length,
    #                                          temInput_length=temInput_length,
    #                                          depth_spec=depth_spec, depth_tem=depth_tem, gr_spec=gr_spec, gr_tem=gr_tem,
    #                                          nb_dense_block=nb_dense_block, nb_class=nb_class).cuda(1)

    sst = SST(input_width=input_width, specInput_length=specInput_length,
                                             temInput_length=temInput_length,
                                             depth_spec=depth_spec, depth_tem=depth_tem, gr_spec=gr_spec, gr_tem=gr_tem,
                                             nb_dense_block=nb_dense_block, nb_class=nb_class)

    opt = Adam(lr=lr, params=[{"params":sst.parameters()}], weight_decay=1e-5)
    # loss_func = nn.CrossEntropyLoss().cuda(1)
    loss_func = nn.CrossEntropyLoss()

    for i in range(1, 16):
        all_result_file = open(os.path.join(result_path, 'all_result.txt'), "a")
        print('Subject:' + str(i))
        print('Subject ' + str(i) + ":", file=all_result_file)
        all_result_file.close()
        for j in range(1, 4):
            print("  Session:" + str(j))
            train_specInput = np.load(os.path.join(
                train_specInput_root_path, f"subject_{i}/section_{j}_data.npy"))
            train_tempInput = np.load(os.path.join(
                train_tempInput_root_path, f"subject_{i}/section_{j}_data.npy"))
            train_label = np.load(os.path.join(
                train_label_root_path, f"subject_{i}/section_{j}_data.npy"))

            index = np.arange(train_specInput.shape[0])
            np.random.shuffle(index)

            train_specInput = train_specInput[index]
            train_tempInput = train_tempInput[index]
            train_label = train_label[index]

            train_label = [x + 1 for x in train_label]
            # train_label = to_categorical(train_label, num_classes=3)

            # Evaluate
            test_specInput = np.load(os.path.join(
                test_specInput_root_path, f"subject_{i}/section_{j}_data.npy"))
            test_tempInput = np.load(os.path.join(
                test_tempInput_root_path, f"subject_{i}/section_{j}_data.npy"))

            test_label = np.load(os.path.join(
                test_label_root_path, f"subject_{i}/section_{j}_data.npy"))

            test_label = [x + 1 for x in test_label]
            # test_label = to_categorical(test_label, num_classes=3)

            train_specInput = to_tensor(train_specInput)
            train_tempInput = to_tensor(train_tempInput)
            train_label = to_tensor(train_label, False)

            test_specInput = to_tensor(test_specInput)
            test_tempInput = to_tensor(test_tempInput)
            test_label = to_tensor(test_label, False)

            train(train_specInput, train_tempInput, train_label, sst, loss_func, opt)
            accuracy = evaluate(test_specInput, test_tempInput, test_label, sst, loss_func)

            all_result_file = open(os.path.join(result_path, 'all_result.txt'), "a")
            print('  Session ' + str(j) + ":" +
                  str(accuracy), file=all_result_file)
            all_result_file.close()

if __name__ == '__main__':
    read_config('./config/SEED.ini')
    run()