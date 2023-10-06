import os

RES = [126, 224, 364, 504]
for res_h in RES:
    for res_w in RES:
        os.system(
            f'python3 scripts/convert-to-onnx.py --image_height {res_h} --image_width {res_w} --batch_size 1 --opset_version 15'
        )
