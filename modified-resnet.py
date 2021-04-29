from mxnet import nd
from mxnet.gluon import nn
import d2lzh as d2l
import mxnet as mx
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
from mxnet import autograd, nd
from mxnet import nd
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from mxnet.io import ImageRecordIter
import os, time, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model


from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys

import os
from mxnet import nd
from mxnet.io import ImageRecordIter

rec_path = os.path.expanduser('/rds/project/t2_vol4/rds-t2-cspp011/CT/rec/')

# You need to specify ``root`` for ImageNet if you extracted the images into
# a different folder
train_data = ImageRecordIter(
        path_imgrec = os.path.join(rec_path, 'Train_rec.rec'),
        path_imgidx = os.path.join(rec_path, 'Train_rec.idx'),
        resize = 512,
        data_shape  = (3, 512, 512),
        batch_size  = 128,  
        preprocess_threads = 8,
        shuffle = True,
    )

from gluoncv.utils import viz

test_data = ImageRecordIter(
    path_imgrec = os.path.join(rec_path, 'Test_rec.rec'),
    path_imgidx = os.path.join(rec_path, 'Test_rec.idx'),
    resize = 512,
    data_shape  = (3, 512, 512),
    batch_size  = 128, 
    preprocess_threads = 8,
    shuffle = False,
)

val_data = ImageRecordIter(
    path_imgrec = os.path.join(rec_path, 'Val_rec.rec'),
    path_imgidx = os.path.join(rec_path, 'Val_rec.idx'),
    resize = 512,
    data_shape  = (3, 512, 512),
    batch_size  = 64, 
    preprocess_threads = 8,
    shuffle = False
)
#for batch in val_data:
#    viz.plot_image(nd.transpose(batch.data[0][0], (1, 2, 0)))
#    print(batch.label[0][0])
#    print(batch.label)
    
    
class DataIterLoader():
  def __init__(self, data_iter):
    self.data_iter = data_iter

  def __iter__(self):
    self.data_iter.reset()
    return self

  def __next__(self):
    batch = self.data_iter.__next__()
    assert len(batch.data) == len(batch.label) == 1
    data = batch.data[0]
    label = batch.label[0]
    return data, label

  #def next(self):
  #  return self.__next__()  # for Python 2

train_iter = DataIterLoader(train_data)
test_iter = DataIterLoader(test_data)
val_iter = DataIterLoader(val_data)

def try_all_gpus():  
    ctxes = []
    try:
        for i in range(16):  
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes

ctx = try_all_gpus()

def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])
    
def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    global hold_accuracy_train
    global hold_accuracy_val
    hold_accuracy_train = []    
    hold_accuracy_val = []
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        hold_accuracy_val.append(test_acc)
     #   train_acc = evaluate_accuracy(train_iter, net, ctx)
        hold_accuracy_train.append(train_acc_sum / m)
        print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
                 time.time() - start))
        if test_acc >= 0.993:
            break

loss = gloss.SoftmaxCrossEntropyLoss()
model_name = 'resnet18_v1'
classes = 3

def conv_block(num_channels):
    blk = nn.HybridSequential() 
    blk.add(nn.Conv2D(num_channels,(1,1)),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.GlobalAvgPool2D())
    return blk


class modifiedMobileNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(modifiedMobileNet, self).__init__(**kwargs)
        finetune_net = get_model(model_name, pretrained=True)
        with self.name_scope():
           self.part1 = finetune_net.features[:4]
           self.part2 = finetune_net.features[4]
           self.part3 = finetune_net.features[5]
           self.part4 = finetune_net.features[6]
           self.part5 = finetune_net.features[7:]
           self.part6 = nn.Dense(classes)
           self.flatten = nn.Flatten()
           self.convblk1 = conv_block(32) #16
           self.convblk2 = conv_block(32) #16
           self.convblk3 = conv_block(32) #16
           self.convblk4 = conv_block(32) #16
           self.convblk5 = conv_block(32) #16
           self.w = conv_block(5)
           

    def forward(self,x):
        w = self.w(x)
        w = self.flatten(w)
        x = self.part1(x)
        x1 = self.convblk1(x)
        x = self.part2(x)
        x2 = self.convblk2(x)
        x = self.part3(x)
        x3 = self.convblk3(x)
        x = self.part4(x)
        x4 = self.convblk4(x)
        x = self.part5(x)
        x5 = self.convblk5(x)
        x = w[0]*x1+w[1]*x2+w[2]*x3+w[3]*x4+w[4]*x5
    #    x = nd.concat(w[0]*x, w[1]*x1, w[2]*x2, w[3]*x3, w[4]*x4, dim=1)
        x = self.flatten(x)
        x = self.part6(x)
        return x

net1 = modifiedMobileNet()      
net1.collect_params('.*modifiedmobilenet').initialize(init.Xavier(), ctx = ctx)
net1.collect_params().reset_ctx(ctx)
net1.hybridize()
epochs = 25 #20
trainer1 = gluon.Trainer(net1.collect_params(), 'adam') #{'learning_rate': 0.1, 'wd': 1e-6}
train(train_iter, test_iter, net1, loss, trainer1, ctx, num_epochs=epochs)
final_test_acc = evaluate_accuracy(test_iter, net1, ctx)
print('resnet18_w32_in512:', final_test_acc)

np.save('/rds/project/t2_vol4/rds-t2-cspp011/CT/results/resnet18_w32_in512_train.npy', hold_accuracy_train)
np.save('/rds/project/t2_vol4/rds-t2-cspp011/CT/results/resnet18_w32_in512_test.npy', hold_accuracy_val)
file_name = "/rds/project/t2_vol4/rds-t2-cspp011/CT/models/resnet-w32-input512.params"
net1.save_parameters(file_name)