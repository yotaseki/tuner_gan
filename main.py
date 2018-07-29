#!/usr/bin/env python
import os
import sys

import numpy as np
import chainer
from chainer import training,datasets, iterators
from chainer.training import extension
from chainer.training import extensions

from updater import Updater
from net import Discriminator,Generator
from evaluation import sample_generate, sample_generate_light
#from record import record_setting


def setup_adam_optimizer(model):
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    return optimizer

def main():
    gpu_id = 0
    batchsize = 10
    report_keys = ["loss_dis", "loss_gen"]
    
    train_dataset = datasets.LabeledImageDataset(str(argv[1]))
    train_iter = iterators.SerialIterator(train_dataset,batchsize)
    
    models = []
    generator = Generator()
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
        "iterator": {'main': train_iter},
        "device": gpu_id,
        "optimizer": opts,
        "models": models
    }
    output = 'result'
    display_interval = 20
    evaluation_interval = 1000
    max_iter = 10000
    
    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=output)
    trainer.extend(extensions.LogReport(keys=report_keys,trigger=(display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(display_interval, 'iteration'))
    trainer.extend(sample_generate(generator, output), trigger=(evaluation_interval, 'iteration'), priority=extension.PRIORITY_WRITER)
    trainer.run()

if __name__ == '__main__':
    argv = sys.argv
    main()
