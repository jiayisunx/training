# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from torch.utils import mkldnn as mkldnn_utils
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, log_path, timer=None):
    model.eval()
    if os.environ.get('USE_MKLDNN') == "1" and os.environ.get('USE_BF16') != "1":
        model = mkldnn_utils.to_mkldnn(model)
    results_dict = {}
    cpu_device = torch.device("cpu")

    if os.environ.get('PROFILE') == "1":
        for i, batch in enumerate(tqdm(data_loader)):
            with torch.autograd.profiler.profile() as prof:
                images, targets, image_ids = batch
                images = images.to(device)
                with torch.no_grad():
                    output = model(images)
                    output = [o.to(cpu_device) for o in output]
                results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
            prof.export_chrome_trace(os.path.join(log_path, 'result_' + str(i) + '.json'))
    else:
        for i, batch in enumerate(tqdm(data_loader)):
            images, targets, image_ids = batch
            images = images.to(device)
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = model(images)
                if timer:
                    if not device.type == 'cpu':
                        torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        log_path='./log/',
        warmup=0,
        performance_only=False
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    if hasattr(data_loader.batch_sampler, "batch_sampler"):
        batch_size = data_loader.batch_sampler.batch_sampler.batch_size
    else:
        batch_size = data_loader.batch_sampler.batch_size
    logger.info("Start evaluation on {} dataset({} iterations).".format(dataset_name, len(data_loader)))
    total_timer = Timer()
    inference_timer = Timer(warmup)
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, log_path, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc(average=False)
    total_time_str = get_time_str(total_time)
    totle_imgs = batch_size * (len(data_loader) - warmup)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time / totle_imgs, num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time / totle_imgs,
            num_devices,
        )
    )
    logger.info(
        "Model inference performance: {} imgs / s per device, on {} devices".format(
            totle_imgs / inference_timer.total_time,
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process() or performance_only:
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
