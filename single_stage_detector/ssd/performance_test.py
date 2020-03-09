import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
from ssd_r34 import SSD_R34
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import random
import numpy as np
from mlperf_compliance import mlperf_log
from mlperf_logger import ssd_print, broadcast_seeds
from torch.utils import mkldnn as mkldnn_utils

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--arch', '-a', default='ssd300',
                        help='architectures: ssd300, ssd_r34')
    parser.add_argument('--dummy', type=int, default=0,
                        help='use dummy data')
    parser.add_argument('--log', type=str, default='./',
                        help='folder to save profiling result')
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--num-workers', '-j', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=0.23,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--totle-iteration', type=int, default=0,
                        help='iteration to run performance test, 0 means no limited')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--image_size', default=[300,300], type=int, nargs='+',
                        help='input image sizes (e.g 300 300,1200 1200)')
    parser.add_argument('--strides', default=[3,3,2,2,2,2], type=int, nargs='+',
                        help='stides for ssd model must include 6 numbers')  
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--evaluation', nargs='*', type=int,
                        default=[40, 50, 55, 60, 65, 70, 75, 80],
                        help='epochs at which to evaluate')
    parser.add_argument('-m', '--mode', metavar='MODE', default='training',
                    choices=["training", "inference"],
                    help='running mode: ' +
                        ' | '.join(["training", "inference"]) +
                        ' (default: training)')
    parser.add_argument('--lr-decay-schedule', nargs='*', type=int,
                        default=[40, 50],
                        help='epochs at which to decay the learning rate')
    parser.add_argument('--warmup', type=float, default=None,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--warmup-factor', type=int, default=0,
                        help='mlperf rule parameter for controlling warmup curve')
    parser.add_argument('--perf-prerun-warmup', type=int, default=0,
                        help='how much iterations to pre run before performance test, -1 mean use all dataset.')
    parser.add_argument('--lr', type=float, default=2.5e-3,
                        help='base learning rate')
    # Distributed stuff
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')

    return parser.parse_args()


def save_time(file_name, content):
    import json
    #file_name = os.path.join(args.log, 'result_' + str(idx) + '.json')
    finally_result = []
    with open(file_name, "r",encoding="utf-8") as f:
        data = json.loads(f.read())
        finally_result += data
        finally_result += content
        with open(file_name,"w",encoding="utf-8") as f:
            f.write(json.dumps(finally_result,ensure_ascii=False,indent=2))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes_R34_coco(figsize,strides):
    ssd_r34=SSD_R34(81,strides=strides)
    synt_img=torch.rand([1,3]+figsize)
    _,_,feat_size =ssd_r34(synt_img, extract_shapes = True)
    print('Features size: ', feat_size)
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] 
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    ssd_print(key=mlperf_log.FEATURE_SIZES, value=feat_size)

    steps = [8, 16, 32, 64, 100, 300]
    ssd_print(key=mlperf_log.STEPS, value=steps)

    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    ssd_print(key=mlperf_log.SCALES, value=scales)

    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    ssd_print(key=mlperf_log.ASPECT_RATIOS, value=aspect_ratios)

    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    ssd_print(key=mlperf_log.NUM_DEFAULTS,
                         value=len(dboxes.default_boxes))
    return dboxes


def coco_eval(model, coco, cocoGt, encoder, inv_map,
              epoch, iteration, args, use_cuda=True, is_inference_test=False):
    from pycocotools.cocoeval import COCOeval

    batch_time = AverageMeter('Time', ':6.3f')

    print("")
    model.eval()
    if os.environ.get('USE_MKLDNN') == "1":
        model = mkldnn_utils.to_mkldnn(model)
    if use_cuda:
        model.cuda()
    ret = []

    overlap_threshold = 0.50
    nms_max_detections = 200
    ssd_print(key=mlperf_log.NMS_THRESHOLD,
                         value=overlap_threshold, sync=False)
    ssd_print(key=mlperf_log.NMS_MAX_DETECTIONS,
                         value=nms_max_detections, sync=False)

    ssd_print(key=mlperf_log.EVAL_START, value=epoch, sync=False)

    if args.dummy: 
        img_keys = [1] * 5000
        img = torch.randn(3, args.image_size[0], args.image_size[1])
    else:
        img_keys = coco.img_keys

    totle_iter = args.totle_iteration if args.totle_iteration > 0 else (len(coco) - args.perf_prerun_warmup)
    assert totle_iter >= args.perf_prerun_warmup
    if os.environ.get('PROFILE') == "1" and is_inference_test:
        with torch.autograd.profiler.profile() as prof:
            for idx, image_id in enumerate(coco.img_keys):
                if not args.dummy:
                    img, (htot, wtot), _, _ = coco[idx]

                with torch.no_grad():
                    print("Parsing image: {}/{}".format(idx+1, totle_iter + args.perf_prerun_warmup))
                    inp = img.unsqueeze(0)
                    if os.environ.get('USE_MKLDNN') == "1":
                        inp = inp.to_mkldnn()
                    if use_cuda:
                        inp = inp.cuda()

                    if args.perf_prerun_warmup > 0 and idx >= args.perf_prerun_warmup:
                        start_time=time.time()

                    if 'ssd_r34' == args.arch:
                        ploc, plabel, _ = model(inp)
                    else:
                        ploc, plabel = model(inp)

                    try:
                        result = encoder.decode_batch(ploc, plabel,
                                                      overlap_threshold,
                                                      nms_max_detections)[0]

                    except:
                        #raise
                        print("")
                        print("No object detected in idx: {}".format(idx))
                        continue
                    finally:
                        if args.perf_prerun_warmup > 0 and idx >= args.perf_prerun_warmup:
                            batch_time.update(time.time()-start_time)

                    if args.dummy:
                        continue

                    loc, label, prob = [r.cpu().numpy() for r in result]
                    for loc_, label_, prob_ in zip(loc, label, prob):
                        ret.append([image_id, loc_[0]*wtot, \
                                              loc_[1]*htot,
                                              (loc_[2] - loc_[0])*wtot,
                                              (loc_[3] - loc_[1])*htot,
                                              prob_,
                                              inv_map[label_]])
                if args.totle_iteration > 0 and idx >= (args.totle_iteration + args.perf_prerun_warmup - 1):
                    break
        prof.export_chrome_trace(os.path.join(args.log, "result.json"))
    else:
        start = time.time()
        for idx, image_id in enumerate(coco.img_keys):
            if not args.dummy:
                img, (htot, wtot), _, _ = coco[idx]

            with torch.no_grad():
                print("Parsing image: {}/{}".format(idx+1, totle_iter + args.perf_prerun_warmup))
                inp = img.unsqueeze(0)
                if os.environ.get('USE_MKLDNN') == "1":
                    inp = inp.to_mkldnn()
                if use_cuda:
                    inp = inp.cuda()

                start_time=time.time()

                if os.environ.get('PROFILE_ITER') == "1" and is_inference_test:
                    with torch.autograd.profiler.profile() as prof:
                        if 'ssd_r34' == args.arch:
                            ploc, plabel, _ = model(inp)
                        else:
                            ploc, plabel = model(inp)
                    prof.export_chrome_trace(os.path.join(args.log, 'result_' + str(idx) + '.json'))
                    mode_inference = time.time()-start_time
                    print('Mode inference time: ', time.time()-start_time)
                    file_name = os.path.join(args.log, 'result_' + str(idx) + '.json')
                    content = [{"inference_time": mode_inference}]
                    save_time(file_name, content)
                    start_decode = time.time()
                else:
                    if 'ssd_r34' == args.arch:
                        ploc, plabel, _ = model(inp)
                    else:
                        ploc, plabel = model(inp)

                try:
                    result = encoder.decode_batch(ploc, plabel,
                                                  overlap_threshold,
                                                  nms_max_detections)[0]
                except:
                    #raise
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue
                finally:
                    if args.perf_prerun_warmup > 0 and idx >= args.perf_prerun_warmup:
                        batch_time.update(time.time()-start_time)
                    if is_inference_test and args.totle_iteration > 0 and idx >= (args.totle_iteration + args.perf_prerun_warmup - 1):
                        break
                
                if os.environ.get('PROFILE_ITER') == "1" and is_inference_test:
                    decoding_time = time.time()-start_decode
                    file_name = os.path.join(args.log, 'result_' + str(idx) + '.json')
                    content = [{"decoding_time": decoding_time}]
                    save_time(file_name, content)
                if args.dummy:
                    continue

                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([image_id, loc_[0]*wtot, \
                                          loc_[1]*htot,
                                          (loc_[2] - loc_[0])*wtot,
                                          (loc_[3] - loc_[1])*htot,
                                          prob_,
                                          inv_map[label_]])
    print("")
    if is_inference_test:
        latency = batch_time.sum / totle_iter * 1000
        perf = totle_iter / batch_time.sum
        print('inference latency %3.0f ms'%latency)
        print('inference performance %3.0f fps'%perf)
        return True
    else:
        print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    if len(ret) > 0 and not args.dummy:
        cocoDt = cocoGt.loadRes(np.array(ret))

        E = COCOeval(cocoGt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

        # put your model back into training mode
        model.train()

        current_accuracy = E.stats[0]
        ssd_print(key=mlperf_log.EVAL_SIZE, value=idx + 1, sync=False)
        ssd_print(key=mlperf_log.EVAL_ACCURACY,
                             value={"epoch": epoch,
                                    "value": current_accuracy},
                  sync=False)
        ssd_print(key=mlperf_log.EVAL_ITERATION_ACCURACY,
                             value={"iteration": iteration,
                                    "value": current_accuracy},
                  sync=False)
        ssd_print(key=mlperf_log.EVAL_TARGET, value=threshold, sync=False)
        ssd_print(key=mlperf_log.EVAL_STOP, value=epoch, sync=False)
        return current_accuracy>= args.threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
    else:
        return []


def eval300_mlperf_coco(args):
    global torch
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    start_time=time.time()
    if 'ssd_r34' == args.arch:
        dboxes = dboxes_R34_coco(args.image_size,args.strides)
    else:
        dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    val_trans = SSDTransformer(dboxes, (args.image_size[0], args.image_size[1]), val=True)

    if args.dummy:
        val_coco = COCODetection(None, None, val_trans)
        inv_map=[]
    else:
        val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(args.data, "val2017")

        cocoGt = COCO(annotation_file=val_annotate)
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
        inv_map = {v:k for k,v in val_coco.label_map.items()}

    labelnum = val_coco.labelnum

    # if args.distributed:
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_coco)
    # else:
    #     val_sampler = None
    # val_dataloader = DataLoader(val_coco,
    #                             batch_size=args.batch_size,
    #                             shuffle=False,
    #                             sampler=val_sampler,
    #                             num_workers=args.num_workers)
    preprocess_time = time.time()-start_time
    print('preprocessing time: ', preprocess_time)
    if 'ssd_r34' == args.arch:
        ssd = SSD_R34(labelnum,strides=args.strides)
    else:
        ssd = SSD300(labelnum)

    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd.load_state_dict(od["model"])

    if use_cuda:
        ssd.cuda()

    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda()

    return coco_eval(ssd, val_coco, cocoGt, encoder, inv_map, 0, 0, args, use_cuda, is_inference_test=True)


def lr_warmup(optim, wb, iter_num, base_lr, args):
    if iter_num < wb:
        # mlperf warmup rule
        warmup_step = base_lr / (wb * (2 ** args.warmup_factor))
        new_lr = base_lr - (wb - iter_num) * warmup_step
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def combine(json_file_list, finally_file_path):
    import json
    finally_result = []
    for file in json_file_list:
        with open(file, "r",encoding="utf-8") as f:
            data = json.loads(f.read())
            finally_result += data
    with open(finally_file_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(finally_result,ensure_ascii=False,indent=2))  

def train300_mlperf_coco(args):
    global torch
    from coco import COCO

    batch_time = AverageMeter('Time', ':6.3f')

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.distributed = False
    if use_cuda:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
        except:
            raise ImportError("Please install APEX from https://github.com/nvidia/apex")

    if args.distributed:
        # necessary pytorch imports
        import torch.utils.data.distributed
        import torch.distributed as dist
 #     ssd_print(key=mlperf_log.RUN_SET_RANDOM_SEED)
        if args.no_cuda:
            device = torch.device('cpu')
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda')
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            # set seeds properly
            args.seed = broadcast_seeds(args.seed, device)
            local_seed = (args.seed + dist.get_rank()) % 2**32
            print(dist.get_rank(), "Using seed = {}".format(local_seed))
            torch.manual_seed(local_seed)
            np.random.seed(seed=local_seed)


    start_time=time.time()
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    input_size = 300
    train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False)
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
    ssd_print(key=mlperf_log.INPUT_SIZE, value=input_size)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    if args.dummy:
        val_coco = COCODetection(None, None, val_trans)
        train_coco = COCODetection(None, None, train_trans)
    else:
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
        train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
        cocoGt = COCO(annotation_file=val_annotate)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_coco)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_coco)
    else:
        train_sampler = None
        val_sampler = None
    train_dataloader = DataLoader(train_coco,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_coco,
                                  batch_size=1,
                                  shuffle=False,
                                  sampler=val_sampler,
                                  num_workers=args.num_workers)
    preprocess_time = time.time()-start_time
    preprocess_time = preprocess_time / len(train_coco) 

    # set shuffle=True in DataLoader
    ssd_print(key=mlperf_log.INPUT_SHARD, value=None)
    ssd_print(key=mlperf_log.INPUT_ORDER)
    ssd_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)

    if 'ssd_r34' == args.arch:
        ssd = SSD_R34(train_coco.labelnum)
    else:
        ssd = SSD300(train_coco.labelnum)

    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd.load_state_dict(od["model"])

    ssd.train()
    if use_cuda:
        ssd.cuda()
    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda()
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    # parallelize
    if args.distributed:
        ssd = DDP(ssd)

    global_batch_size = N_gpu * args.batch_size
    current_lr = args.lr * (global_batch_size / 32)
    current_momentum = 0.9
    current_weight_decay = 5e-4
    optim = torch.optim.SGD(ssd.parameters(), lr=current_lr,
                            momentum=current_momentum,
                            weight_decay=current_weight_decay)
    ssd_print(key=mlperf_log.OPT_NAME, value="SGD")
    ssd_print(key=mlperf_log.OPT_LR, value=current_lr)
    ssd_print(key=mlperf_log.OPT_MOMENTUM, value=current_momentum)
    ssd_print(key=mlperf_log.OPT_WEIGHT_DECAY,
                         value=current_weight_decay)
    eval_points = args.evaluation
    print("epoch", "nbatch", "loss")

    iter_num = args.iteration
    avg_loss = 0.0
    inv_map=[]
    if not args.dummy:
        inv_map = {v:k for k,v in val_coco.label_map.items()}
    success = torch.zeros(1)
    if use_cuda:
        success = success.cuda()


    if args.warmup:
        nonempty_imgs = len(train_coco)
        wb = int(args.warmup * nonempty_imgs / (N_gpu*args.batch_size))
        warmup_step = lambda iter_num, current_lr: lr_warmup(optim, wb, iter_num, current_lr, args)
    else:
        warmup_step = lambda iter_num, current_lr: None

    for epoch in range(args.epochs):
        ssd_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        # set the epoch for the sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch in args.lr_decay_schedule:
            current_lr *= 0.1
            print("")
            print("lr decay step #{num}".format(num=args.lr_decay_schedule.index(epoch) + 1))
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr
            ssd_print(key=mlperf_log.OPT_LR,
                                 value=current_lr)

        if os.environ.get('PROFILE') == "1":
            with torch.autograd.profiler.profile() as prof:
                for nbatch, (img, img_size, bbox, label) in enumerate(train_dataloader):
                    if os.environ.get('USE_MKLDNN') == "1":
                        img = img.to_mkldnn()
                    if use_cuda:
                        img = img.cuda()
                    img = Variable(img, requires_grad=False)
                    start = time.time()
                    if 'ssd_r34' == args.arch:
                        ploc, plabel, _ = ssd(img)
                    else:
                        ploc, plabel = ssd(img)
                    trans_bbox = bbox.transpose(1,2).contiguous()
                    if use_cuda:
                        trans_bbox = trans_bbox.cuda()
                        label = label.cuda()
                    gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                                   Variable(label, requires_grad=False)
                    loss = loss_func(ploc, plabel, gloc, glabel)

                    optim.zero_grad()
                    loss.backward()
                    warmup_step(iter_num, current_lr)
                    optim.step()

                    if (iter_num - args.iteration) >= args.perf_prerun_warmup:
                        batch_time.update(time.time() - start)

                    if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()

                    print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                                .format(iter_num, loss.item(), avg_loss), end="\r")

                    iter_num += 1
                    if args.totle_iteration > 0 and (iter_num - args.iteration) >= args.totle_iteration:
                        break
            prof.export_chrome_trace('result.json')
        else:
            for nbatch, (img, img_size, bbox, label) in enumerate(train_dataloader):
                if os.environ.get('USE_MKLDNN') == "1":
                    img = img.to_mkldnn()
                if use_cuda:
                    img = img.cuda()
                img = Variable(img, requires_grad=False)
                start = time.time()
                if os.environ.get('PROFILE_ITER') == "1":
                    with torch.autograd.profiler.profile() as prof:
                        if 'ssd_r34' == args.arch:
                            ploc, plabel, _ = ssd(img)
                        else:
                            ploc, plabel = ssd(img)
                    prof.export_chrome_trace(os.path.join(args.log, 'result_' + str(nbatch) + '.json'))
                else:
                    if 'ssd_r34' == args.arch:
                        ploc, plabel, _ = ssd(img)
                    else:
                        ploc, plabel = ssd(img)
                trans_bbox = bbox.transpose(1,2).contiguous()
                if use_cuda:
                    trans_bbox = trans_bbox.cuda()
                    label = label.cuda()
                gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                               Variable(label, requires_grad=False)
                loss = loss_func(ploc, plabel, gloc, glabel)

                if os.environ.get('PROFILE_ITER') == "1":
                    optim.zero_grad()
                    with torch.autograd.profiler.profile() as prof:
                        loss.backward()
                    prof.export_chrome_trace(os.path.join(args.log, 'backward_result_' + str(nbatch) + '.json'))
                    warmup_step(iter_num, current_lr)
                    
                    start_time=time.time()
                    optim.step()
                    weights_update_time = time.time()-start_time

                    fwd_file = os.path.join(args.log, 'result_' + str(nbatch) + '.json')
                    bwd_file = os.path.join(args.log, 'backward_result_' + str(nbatch) + '.json')
                    json_file_list = [fwd_file, bwd_file]
                    combine(json_file_list, fwd_file)
                    if (os.path.exists(bwd_file)):
                        os.remove(bwd_file)
                    
                    file_name = os.path.join(args.log, 'result_' + str(nbatch) + '.json')
                    content = [{"weights_update_time": weights_update_time}]
                    save_time(file_name, content)
                else:
                    optim.zero_grad()
                    loss.backward()
                    warmup_step(iter_num, current_lr)
                    optim.step()

                if (iter_num - args.iteration) >= args.perf_prerun_warmup:
                    batch_time.update(time.time() - start)

                if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()

                print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                            .format(iter_num, loss.item(), avg_loss), end="\r")

                iter_num += 1
                if args.totle_iteration > 0 and (iter_num - args.iteration) >= args.totle_iteration:
                    break

        if (epoch + 1 in eval_points) and not args.dummy:
            rank = dist.get_rank() if args.distributed else args.local_rank
            if args.distributed:
                world_size = float(dist.get_world_size())
                for bn_name, bn_buf in ssd300.module.named_buffers(recurse=True):
                    if ('running_mean' in bn_name) or ('running_var' in bn_name):
                        dist.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                        bn_buf /= world_size
            if rank == 0:
                if not args.no_save:
                    print("")
                    print("saving model...")
                    torch.save({"model" : ssd.state_dict(), "label_map": train_coco.label_info},
                               "./models/iter_{}.pt".format(iter_num))

                if coco_eval(ssd, val_coco, cocoGt, encoder, inv_map, epoch + 1,iter_num, args, use_cuda):
                    success = torch.ones(1)
                    if use_cuda:
                        success = success.cuda()
            if args.distributed:
                dist.broadcast(success, 0)
            if success[0]:
                    return True

        batch_size = train_dataloader.batch_size
        latency = batch_time.sum / (iter_num - args.iteration) / batch_size * 1000
        perf = (iter_num - args.iteration) * batch_size/batch_time.sum
        print('training latency %3.0f ms'%latency)
        print('training performance %3.0f fps'%perf)

    return False

def main():
    args = parse_args()

    if args.local_rank == 0:
        if not os.path.isdir('./models'):
            os.mkdir('./models')

    torch.backends.cudnn.benchmark = True

    if args.totle_iteration > 0:
        assert args.totle_iteration > args.perf_prerun_warmup

    # start timing here
    ssd_print(key=mlperf_log.RUN_START)

    if args.mode == "training":
        success = train300_mlperf_coco(args)
    else:
        if args.seed is not None:
            print("Using seed = {}".format(args.seed))
            torch.manual_seed(args.seed)
            np.random.seed(seed=args.seed)
        success = eval300_mlperf_coco(args)

    # end timing here
    ssd_print(key=mlperf_log.RUN_STOP, value={"success": success})
    ssd_print(key=mlperf_log.RUN_FINAL)

if __name__ == "__main__":
    main()
