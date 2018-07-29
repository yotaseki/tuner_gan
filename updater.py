import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp
        
        black_batch = self.get_iterator('main').next()
        white_batch = self.get_iterator('label').next()
        batchsize = len(black_batch)
        xr = []
        xf = []
        for i in range(batchsize):
            xf.append(np.asarray(black_batch[i]).astype("f"))
            xr.append(np.asarray(white_batch[i]).astype("f"))

        x_real = Variable(xp.asarray(xr))
        y_real = self.dis(x_real)

        #z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        #x_fake = self.gen(z)
        x_fv = Variable(xp.asarray(xf))
        x_fake = self.gen(x_fv)
        y_fake = self.dis(x_fake)

        loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        loss_dis += F.sum(F.softplus(y_fake)) / batchsize

        loss_gen = F.sum(F.softplus(-y_fake)) / batchsize

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        x_fake.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_gen': loss_gen})
        chainer.reporter.report({'loss_dis': loss_dis})
