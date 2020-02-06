import os, sys
from time import time

import numpy as np
import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.utils import logging
from slowfast.utils import misc
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.datasets.cv2_transform import scale

logger = logging.get_logger(__name__)
np.random.seed(20)

LABELS_INTERESTED_IN = [51, 57, 63, 65, 69, 70, 72, 75]


class VideoReader(object):

    def __init__(self, cfg):
        self.source = cfg.VIDEO_DEMO.DATA_SOURCE
        self.display_width = cfg.VIDEO_DEMO.DISPLAY_WIDTH
        self.display_height = cfg.VIDEO_DEMO.DISPLAY_HEIGHT

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            self.cap.release()
            raise StopIteration

        return was_read, frame


def video(cfg):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    # Build the video model and print model statistics.
    model = build_model(cfg)
    model.eval()
    misc.log_model_info(model)

    # Load a checkpoint to test if applicable.
    if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    cu.load_checkpoint(ckpt, model, cfg.NUM_GPUS > 1, None, inflation=False,
                       convert_from_caffe2="caffe2" in [cfg.TEST.CHECKPOINT_TYPE, cfg.TRAIN.CHECKPOINT_TYPE])

    # Load object detector from detectron2
    dtron2_cfg_file = cfg.VIDEO_DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_CFG
    dtron2_cfg = get_cfg()
    dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
    dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    dtron2_cfg.MODEL.WEIGHTS = cfg.VIDEO_DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_WEIGHTS
    object_predictor = DefaultPredictor(dtron2_cfg)
    # Load the labels of AVA dataset
    with open(cfg.VIDEO_DEMO.LABEL_FILE_PATH) as f:
        labels = f.read().split('\n')[:-1]
    palette = np.random.randint(64, 128, (len(labels), 3)).tolist()
    boxes = []

    frame_provider = VideoReader(cfg)
    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    frames = []
    pred_labels = []

    # main start
    main_start = time()

    for able_to_read, frame in frame_provider:
        if not able_to_read:
            # when reaches the end frame, clear the buffer and continue to the next one.
            frames = []
            continue

        if len(frames) != seq_len:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
            frames.append(frame_processed)
            if cfg.DETECTION.ENABLE and len(frames) == seq_len // 2 - 1:
                mid_frame = frame

        if len(frames) == seq_len:

            if cfg.DETECTION.ENABLE:
                outputs = object_predictor(mid_frame)
                fields = outputs["instances"]._fields
                pred_classes = fields["pred_classes"]
                selection_mask = pred_classes == 0
                # acquire person boxes
                pred_classes = pred_classes[selection_mask]
                pred_boxes = fields["pred_boxes"].tensor[selection_mask]
                scores = fields["scores"][selection_mask]
                boxes = cv2_transform.scale_boxes(cfg.DATA.TEST_CROP_SIZE,
                                                  pred_boxes,
                                                  frame_provider.display_height,
                                                  frame_provider.display_width)
                boxes = torch.cat([torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1)

            inputs = torch.as_tensor(frames).float()
            inputs = inputs / 255.0
            # Perform color normalization.
            inputs = inputs - torch.tensor(cfg.DATA.MEAN)
            inputs = inputs / torch.tensor(cfg.DATA.STD)
            # T H W C -> C T H W.
            inputs = inputs.permute(3, 0, 1, 2)

            # 1 C T H W.
            inputs = inputs.unsqueeze(0)

            # Sample frames for the fast pathway.
            index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
            fast_pathway = torch.index_select(inputs, 2, index)
            # logger.info('fast_pathway.shape={}'.format(fast_pathway.shape))

            # Sample frames for the slow pathway.
            index = torch.linspace(0, fast_pathway.shape[2] - 1,
                                   fast_pathway.shape[2] // cfg.SLOWFAST.ALPHA).long()
            slow_pathway = torch.index_select(fast_pathway, 2, index)
            # logger.info('slow_pathway.shape={}'.format(slow_pathway.shape))
            inputs = [slow_pathway, fast_pathway]

            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Perform the forward pass.
            # When there is nothing in the scene,
            #   use a dummy variable to disable all computations below.
            if not len(boxes):
                preds = torch.tensor([])
            else:
                preds = model(inputs, boxes)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds = du.all_gather(preds)[0]

            if cfg.DETECTION.ENABLE:
                # if GPU is more powerful, change to CUDA processing.
                preds = preds.cpu().detach().numpy()
                pred_masks = preds > 0.1
                label_ids = [np.nonzero(pred_mask)[0] for pred_mask in pred_masks]

                # filter labels according to LABELS_INTERESTED_IN
                new_label_ids = [[x for x in sublist if x in LABELS_INTERESTED_IN] for sublist in label_ids]

                pred_labels = [
                    [labels[label_id] for label_id in perbox_label_ids]
                    for perbox_label_ids in new_label_ids
                ]

                print(pred_labels)

                # unsure how to detectron2 rescales boxes to image original size, so I use
                # input boxes of slowfast and rescale back it instead, it's safer and even if boxes
                # was not rescaled by cv2_transform.rescale_boxes, it still works.
                # boxes = boxes.cpu().detach().numpy()
                # ratio = np.min(
                #     [frame_provider.display_height, frame_provider.display_width]
                # ) / cfg.DATA.TEST_CROP_SIZE
                # boxes = boxes[:, 1:] * ratio
            else:
                ## Option 1: single label inference selected from the highest probability entry.
                # label_id = preds.argmax(-1).cpu()
                # pred_label = labels[label_id]
                # Option 2: multi-label inferencing selected from probability entries > threshold
                label_ids = torch.nonzero(preds.squeeze() > 0.1).reshape(-1).cpu().detach().numpy()
                pred_labels = labels[label_ids]
                # logger.info(pred_labels)
                if not list(pred_labels):
                    pred_labels = ['Unknown']

            # option 1: remove the oldest frame in the buffer to make place for the new one.
            # frames.pop(0)
            # option 2: empty the buffer
            frames = []

        # if cfg.DETECTION.ENABLE and pred_labels and boxes.any():
        #     for box, box_labels in zip(boxes.astype(int), pred_labels):
        #         cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), thickness=2)
        #         label_origin = box[:2]
        #         for label in box_labels:
        #             label_origin[-1] -= 5
        #             (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
        #             cv2.rectangle(
        #                 frame,
        #                 (label_origin[0], label_origin[1] + 5),
        #                 (label_origin[0] + label_width, label_origin[1] - label_height - 5),
        #                 palette[labels.index(label)], -1
        #             )
        #             cv2.putText(
        #                 frame, label, tuple(label_origin),
        #                 cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        #             )
        #             label_origin[-1] -= label_height + 5



    print('total time: {}'.format(time() - main_start))
