import argparse
import errno
import json
import os
import time

import sys

import numpy as np

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

import torch.nn.functional as F

### Import Data Utils ###
sys.path.append('../')

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

import params

from eval_model import  eval_model

###########################################################
# Comand line arguments, handled by params except seed    #
###########################################################
parser = argparse.ArgumentParser(description='DeepSpeech inference')

parser.add_argument('--model_path', default='models/deepspeech_1.pth',
                    help='Location to save best validation model')

parser.add_argument('--seed', default=0xdeadbeef, type=int, help='Random Seed')

parser.add_argument('--batch_size', default=8, type=int, help='batch size for inference, default is 8')


def to_np(x):
    return x.data.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()

    with open(params.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    decoder = GreedyDecoder(labels)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if params.rnn_type == 'gru' and params.rnn_act_type != 'tanh':
      print("ERROR: GRU does not currently support activations other than tanh")
      sys.exit()

    if params.rnn_type == 'rnn' and params.rnn_act_type != 'relu':
      print("ERROR: We should be using ReLU RNNs")
      sys.exit()

    print("=======================================================")
    for arg in vars(args):
      print("***%s = %s " %  (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")

    criterion = CTCLoss()

    audio_conf = dict(sample_rate=params.sample_rate,
                      window_size=params.window_size,
                      window_stride=params.window_stride,
                      window=params.window,
                      noise_dir=params.noise_dir,
                      noise_prob=params.noise_prob,
                      noise_levels=(params.noise_min, params.noise_max))

    # train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.train_manifest, labels=labels,
    #                                    normalize=True, augment=params.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    # train_loader = AudioDataLoader(train_dataset, batch_size=params.batch_size,
    #                                num_workers=1)
    test_loader = AudioDataLoader(test_dataset, batch_size=params.batch_size,
                                  num_workers=1)

    rnn_type = params.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = torch.load(args.model_path)
    
    avg_loss = 0
    start_epoch = 0
    start_iter = 0
    avg_training_loss = 0

    if params.cuda:
        model         = torch.nn.DataParallel(model).cuda()

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    losses = AverageMeter()
    ctc_time = AverageMeter()   

    model.eval()
    total_cer, total_wer = 0, 0
    inference_time = 0
    end = time.time()

    for i, (data) in enumerate(test_loader):

        inputs, targets, input_percentages, target_sizes = data
        inputs = Variable(inputs, volatile=True)
        # target_sizes = Variable(target_sizes, requires_grad=False)
        # targets = Variable(targets, requires_grad=False)
        
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if params.cuda:
            inputs = inputs.cuda()

        # start inference here
        inf_start = time.time()
        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()

        ctc_start_time = time.time()
        loss = criterion(out, targets, sizes, target_sizes)
        ctc_time.update(time.time() - ctc_start_time)

        loss = loss / inputs.size(0)  # average the loss by minibatch

        loss_sum = loss.data.sum()
        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            print("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.data[0]

        avg_loss += loss_value
        losses.update(loss_value, inputs.size(0))
        
        # end inference here
        inference_time += time.time() - inf_start

        if params.cuda:
            torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # reduce print frequance
        if i % 100 == 0:
            print('Iter: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'CTC Time {ctc_time.val:.3f} ({ctc_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                (i + 1), len(test_loader), batch_time=batch_time,
                ctc_time=ctc_time, loss=losses))

        decoded_output = decoder.decode(out.data, sizes)
        target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
            cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
        total_cer += cer
        total_wer += wer

        del loss
        del out
    
    data_size = params.batch_size * len(test_loader)
    print(' Total time: {time:.3f}\t'
            'Total test_data size {data_size:.3f}\t'.format(
           time=inference_time, data_size=data_size))

    avg_loss /= len(test_loader)
    avg_wer = total_wer / len(test_loader.dataset)
    avg_cer = total_cer / len(test_loader.dataset)

    print('Inference Summary: Average Loss {loss:.3f}\t'
            'Average WER {wer:.3f}\t'
            'Average CER {cer:.3f}\t'.format(
        loss=avg_loss, wer=wer, cer=cer))

if __name__ == '__main__':
    main()