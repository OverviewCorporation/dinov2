"""DINOV2 model converter to onnx."""
import torch
import argparse
import os
import sys
from pathlib import Path
current_path = Path(__file__).resolve()
parent_path = current_path.parent.parent.as_posix()
sys.path.insert(0, parent_path)
import hubconf

from torchvision.transforms import Normalize

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # model will input image in range [0, 255], normalize to imagenet mean and std
        self.norm = Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
        )

    def forward(self, tensor):
        tensor = self.norm(tensor)
        outs = self.model.forward_features(tensor)
        return outs["x_norm_patchtokens"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="dinov2 model name")
parser.add_argument(
    "--image_height", type=int, default=504, help="input image height, must be a multiple of patch_size"
)
parser.add_argument(
    "--image_width", type=int, default=504, help="input image height, must be a multiple of patch_size"
)
parser.add_argument(
    "--patch_size", type=int, default=14, help="dinov2 model patch size, default is 14"
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="dinov2 model batch size, default is 1"
)
parser.add_argument(
    "--opset_version", type=int, default=14, help="onnx opset version, default is 14"
)
parser.add_argument(
    "--output_name", type=str, default='', help="output onnx model name, default is model_name.onnx"
)
args = parser.parse_args()


if __name__ == "__main__":

    assert args.image_height % args.patch_size == 0, f"image height must be multiple of {args.patch_size}, but got {args.image_height}"
    assert args.image_width % args.patch_size == 0, f"image width must be multiple of {args.patch_size}, but got {args.image_height}"

    model = Wrapper(hubconf.dinov2_vits14(for_onnx=True)).to("cpu")
    model.eval()

    dummy_input = torch.rand([args.batch_size, 3, args.image_height, args.image_width]).to("cpu")
    dummy_output = model(dummy_input)
    output_name = (
        args.output_name 
        if args.output_name 
        else f"{args.model_name}_{args.batch_size}-3-{args.image_height}-{args.image_width}.onnx"
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        input_names = ["input"],
        output_names = ["unpooled_features"],
        opset_version=args.opset_version,
        do_constant_folding=True,
    )
