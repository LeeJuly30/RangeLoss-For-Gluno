import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon, autograd

class RangeLossDataLoader(object):
    def __init__(self, dataset, num_class, num_in_class, epochs):
        self._dataset = dataset
        self._num_class = num_class
        self._num_in_class = num_in_class
        self._epochs = epochs
    def __iter__(self):
        unique_label = np.unique(self._dataset[1])
        for _ in range(self._epochs):
            chosen_label_list = np.random.permutation(unique_label)[:self._num_class]
            data = []
            label = []
            for chosen_label in chosen_label_list:
                chosen_index = np.random.permutation(np.where(self._dataset[1]==chosen_label)[0])[:self._num_in_class]
                data.append(self._dataset[0][chosen_index,:,:,:])
                label.append(self._dataset[1][chosen_index])
            data = np.concatenate(data, axis=0)
            label = np.concatenate(label, axis=0)
            yield data, label
    def __len__(self):
        return self._num_class*self._num_in_class

class LeNet(gluon.nn.HybridBlock):
    def __init__(self, classes=10, feature_size=2, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu')
            self.conv2 = gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu')
            self.maxpool = gluon.nn.MaxPool2D(pool_size=2, strides=2)
            self.flat = gluon.nn.Flatten()
            self.dense1 = gluon.nn.Dense(feature_size)
            self.dense2 = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.flat(x)
        ft = self.dense1(x)
        output = self.dense2(ft)

        return output, ft


def _make_conv_block(block_index, num_chan=32, num_layer=2, stride=1, pad=2):
    out = gluon.nn.HybridSequential(prefix='block_%d_' % block_index)
    with out.name_scope():
        for _ in range(num_layer):
            out.add(gluon.nn.Conv2D(num_chan, kernel_size=5, strides=stride, padding=pad))
            out.add(gluon.nn.LeakyReLU(alpha=0.01))
        out.add(gluon.nn.MaxPool2D())

    return out


class LeNetPlus(gluon.nn.HybridBlock):

    def __init__(self, classes=10, feature_size=2, normalize=True, scale=40., **kwargs):
        super(LeNetPlus, self).__init__(**kwargs)
        self.normalize = normalize
        self.scale = scale
        num_chans = [32, 64, 128]
        with self.name_scope():
            self.features = gluon.nn.HybridSequential(prefix='')

            for i, num_chan in enumerate(num_chans):
                self.features.add(_make_conv_block(i, num_chan=num_chan))

            self.features.add(gluon.nn.Dense(feature_size))
            self.features.add(gluon.nn.Flatten())
            self.output = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        ft = self.features(x)
        if self.normalize:
            ft = self.scale*F.broadcast_div(ft, F.expand_dims(F.sqrt(F.sum(F.square(ft), axis=1)), axis=1))
        output = self.output(ft)
        return output, ft
    