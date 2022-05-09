import argparse
import glob
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img

from realbasicvsr.models.builder import build_model

VIDEO_EXTENSIONS = (".mp4", ".mov")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script of RealBasicVSR")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("input_dir", help="directory of the input video")
    parser.add_argument("output_dir", help="directory of the output video")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="maximum sequence length to be processed",
    )
    parser.add_argument(
        "--is_save_as_png", type=bool, default=True, help="whether to save as png"
    )
    parser.add_argument("--fps", type=float, default=25, help="FPS of the output video")
    args = parser.parse_args()

    return args


def init_model(config, checkpoint=None):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.eval()

    return model


def main():
    args = parse_args()

    # initialize the model
    model = init_model(args.config, args.checkpoint)

    # make output dir
    mmcv.mkdir_or_exist(args.output_dir)

    input_paths = sorted(glob.glob(f"{args.input_dir}/*"))

    # map to cuda, if available
    cuda_flag = False
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_flag = True

    for i in range(0, len(input_paths), args.max_seq_len):
        inputs = []
        outputs = []

        # read images
        for input_path in input_paths[i : i + args.max_seq_len]:
            img = mmcv.imread(input_path, channel_order="rgb")
            inputs.append(img)

        # process inputs
        for k, img in enumerate(inputs):
            img = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
            inputs[k] = img.unsqueeze(0)
        inputs = torch.stack(inputs, dim=1)

        # infer
        with torch.no_grad():
            if cuda_flag:
                inputs = inputs.cuda()
            outputs.append(model(inputs, test_mode=True)["output"].cpu())
            outputs = torch.cat(outputs, dim=1)

        # save images
        for k in range(0, outputs.size(1)):
            output = tensor2img(outputs[:, k, :, :, :])
            filename = os.path.basename(input_paths[i + k])
            if args.is_save_as_png:
                file_extension = os.path.splitext(filename)[1]
                filename = filename.replace(file_extension, ".png")
            mmcv.imwrite(output, f"{args.output_dir}/{filename}")


if __name__ == "__main__":
    main()
