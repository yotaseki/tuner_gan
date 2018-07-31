#!/usr/bin/env python
import os
import sys

import numpy as np
from PIL import Image
import chainer
import chainer.functions as F
from chainer import training,datasets, iterators
from chainer.training import extension
from chainer.training import extensions

from updater import Updater
from net import Discriminator,Generator,Filter
from evaluation import sample_filter,sample_generate, sample_generate_light
#from record import record_setting


def setup_adam_optimizer(model):
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    return optimizer

def make_data(img_path):
    img = Image.open(img_path)
    imgArray = np.asarray(img)
    imgArray.flags.writeable = True
    data = np.float32(imgArray)/255.0
    data = F.transpose(data)
    return data

def make_dataset(traintxt):
    dataset = []
    data = open(traintxt)
    for line in data:
        simu, real = line.split()
        #print(simu)
        #print(real)
        s_data = make_data(simu)
        r_data = make_data(real)
        dataset.append([s_data, r_data])
    return dataset

def main():
    gpu_id = 0
    batchsize = 10
    report_keys = ["loss_dis", "loss_gen"]
    
    #train_dataset = datasets.LabeledImageDataset(str(argv[1]))
    black_dataset = datasets.ImageDataset(argv[1])
    white_dataset = datasets.ImageDataset(argv[2])
    #train_dataset = make_dataset(argv[1])
    black_iter = iterators.SerialIterator(black_dataset,batchsize)
    white_iter = iterators.SerialIterator(white_dataset,batchsize)
    #print(np.array(train_dataset[0][1]).shape)
    #print(np.array(train_dataset[1]).shape)
    
    models = []
    #generator = Generator()
    generator = Filter()
    discriminator = Discriminator()
    opts = {}
    opts["opt_gen"] = setup_adam_optimizer(generator)
    opts["opt_dis"] = setup_adam_optimizer(discriminator)
    models = [generator, discriminator]
    chainer.cuda.get_device_from_id(gpu_id).use()
    print("use gpu {}".format(gpu_id))
    for m in models:
        m.to_gpu()
    updater_args = {
        "iterator": {'main': black_iter, 'label': white_iter},
        "device": gpu_id,
        "optimizer": opts,
        "models": models
    }
    output = 'result'
    display_interval = 10
    evaluation_interval = 20
    max_iter = 500
    
    x = []
    x.append(black_dataset[0])
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=output)
    trainer.extend(extensions.LogReport(keys=report_keys,trigger=(display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(display_interval, 'iteration'))
    trainer.extend(sample_filter(generator, x, output), trigger=(evaluation_interval, 'iteration'), priority=extension.PRIORITY_WRITER)
    trainer.run()

if __name__ == '__main__':
    argv = sys.argv
    main()
