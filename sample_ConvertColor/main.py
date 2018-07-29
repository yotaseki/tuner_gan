#!/usr/bin/env python
import os
import sys

import numpy as np
import chainer
from chainer import cuda, training, datasets, iterators
from chainer.training import extension
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
from chainer import Variable

from chainer.cuda import to_cpu
from PIL import Image

class Filter(chainer.Chain):
    def __init__(self, wscale=0.02):
        #w = chainer.initializers.Normal(wscale)
        w = 1.0
        super(Filter, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, 3, 1, 1, 0, nobias=True, initialW=w)

    def __call__(self, x):
        h = self.c0_0(x)
        return h

def main():
    infer_net = Filter()
    gpu_id = 0
    infer_net.to_gpu(gpu_id)
    
    test_data = datasets.LabeledImageDataset(argv[1])
    x = test_data[200][0]
    print(test_data[200][0])
    x = x[None, ]
    x = infer_net.xp.asarray(x)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)
    y = to_cpu(y.array)
    #print(y)
    y = y[0]
    y = np.array(np.clip(y * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, H, W = y.shape
    y = F.transpose(y)
    im = Image.fromarray(y.data)
    im.show()


if __name__ == '__main__':
    argv = sys.argv
    main()
