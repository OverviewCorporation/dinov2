
import os

IMAGE_HEIGHT_DIMS = [126, 224, 364, 504]
IMAGE_WIDTH_DIMS = [126, 224, 364, 504]
BATCH_SIZE = 1
OPSET_VERSION = 15

for img_height in IMAGE_HEIGHT_DIMS:
    for img_width in IMAGE_WIDTH_DIMS:
        os.system(f'python3 scripts/convert-to-onnx.py --image_height {img_height} --image_width {img_width} --batch_size {BATCH_SIZE} --opset_version {OPSET_VERSION}')
