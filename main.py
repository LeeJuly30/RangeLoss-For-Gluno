# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(LeeFlow)s
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon, autograd
import matplotlib.pyplot as plt
from RangeLossForGluon import RangeLoss
from DataLoader import LeNetPlus, RangeLossDataLoader
import seaborn as sns
from utils import evaluate_accuracy, plot_features
import argparse
plt.style.use('ggplot')

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)).asnumpy()  / 255, label.astype(np.int32)

def train():
    mnist_set = gluon.data.vision.MNIST(train=True, transform=transform)
    test_mnist_set = gluon.data.vision.MNIST(train=False, transform=transform)
    data = []
    label = []
    for i in range(len(mnist_set)):
        data.append(mnist_set[i][0][np.newaxis,:,:,:])
        label.append(mnist_set[i][1][np.newaxis,])
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)
    full_set = (data, label)
    ctx = mx.gpu(0)
    model = LeNetPlus(normalize=arg.normalize)
    model.hybridize()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    train_iter = RangeLossDataLoader(full_set, arg.num_class, arg.num_in_class, 15000)
    test_iter = mx.gluon.data.DataLoader(test_mnist_set, 500, shuffle=False)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    Range_loss = RangeLoss(arg.alpha, arg.beta, arg.topk, arg.num_class, arg.num_in_class, 2, arg.margin)
    Range_loss.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                                optimizer='adam', optimizer_params={'learning_rate': arg.lr, 'wd': 5e-4})
    for i, (data, label) in enumerate(train_iter):
        data = nd.array(data, ctx=ctx)
        label = nd.array(label, ctx=ctx)
        with autograd.record():
            output, features = model(data)
            softmax_loss = softmax_cross_entropy(output, label)
            range_loss = Range_loss(features, label)
            loss = softmax_loss + range_loss
        loss.backward()
        trainer.step(data.shape[0])
        if ((i+1)%3000 == 0):
            test_accuracy, test_ft, _, test_lb = evaluate_accuracy(test_iter, model, ctx)
            print(test_accuracy)
            plot_features(test_ft, test_lb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convolutional Neural Networks')
    # File related
    parser.add_argument('--normalize', default=False, type=bool, help='whether or not normalize features')
    parser.add_argument('--alpha', default=1e-2, type=float, help='weight of inter class loss')
    parser.add_argument('--beta', default=1e-2, type=float, help='weight of intra class loss')
    parser.add_argument('--num_class', default=8, type=int, help='numbers of class in a mini-batch')
    parser.add_argument('--num_in_class', default=25, type=int, help='numbers of example in each class ')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--margin', default=18, type=float, help='margin in inter class loss')
    parser.add_argument('--topk', default=2, type=int, help='numbers of top k distance intra class')
    arg = parser.parse_args()
    train()
    