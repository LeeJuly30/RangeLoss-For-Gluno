import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon, autograd

class RangeLoss(gluon.nn.HybridBlock):
    def __init__(self, alpha, beta, top_k, num_class, num_in_class, feature_dim, margin, **kwargs):
        super(RangeLoss, self).__init__(**kwargs)
        self._alpha = alpha
        self._beta = beta
        self._top_k = top_k
        self._num_class = num_class
        self._num_in_class = num_in_class
        self._magrin = margin
    def _pair_distance(self, F, features):
        dot_product = F.dot(features, features.T)
        square_norm = F.sum(F.square(features), axis=1)
        distances = F.expand_dims(square_norm, 0) - 2.0 * dot_product + F.expand_dims(square_norm, 1)
        distances = F.maximum(distances, 0.0)
        mask = F.equal(distances, 0.0)
        distances = distances + mask * 1e-16
        distances = F.sqrt(distances)
        distances = distances * (1.0 - mask)
        return distances
    def _inter_class_loss(self, F, x, y):
        reshape_out = x.reshape((self._num_class,self._num_in_class,-1))
        centers = F.mean(reshape_out, axis=1)
        center_distance = self._pair_distance(F, centers)
        mask = F.array(1.- np.greater_equal.outer(np.arange(self._num_class), np.arange(self._num_class)).astype(np.float32))
        center_distance = center_distance*mask + (1.- mask)*1e4
        center_distance = center_distance.reshape((-1,))
        inter_class_loss = F.maximum(self._magrin - F.min(center_distance), 0)
        return inter_class_loss
    def _intra_class_loss(self, F, x, y):
        intra_class_loss = F.array([0.])
        for i in range(self._num_class):
            same_label_feature = x[i*self._num_in_class:(i+1)*self._num_in_class,:]
            same_label_distance = self._pair_distance(F, same_label_feature)
            mask = F.array(1.- np.greater_equal.outer(np.arange(self._num_in_class), np.arange(self._num_in_class)).astype(np.float32))
            same_label_distance = same_label_distance*mask 
            same_label_distance = same_label_distance.reshape((-1,))
            top_k_distance = F.topk(same_label_distance, k=self._top_k, ret_typ='value', is_ascend=False)
            harmonic_mean = self._top_k/F.sum(1/(top_k_distance+1e-8))
            intra_class_loss  = intra_class_loss + harmonic_mean
        return intra_class_loss
    def hybrid_forward(self, F, x, y):
        inter_class_loss = self._inter_class_loss(F, x, y)
        intra_class_loss = self._intra_class_loss(F, x, y)
        range_loss = self._alpha*inter_class_loss+self._beta*intra_class_loss
        return range_loss