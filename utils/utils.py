import os

import requests
from tqdm import tqdm
import shutil

from PIL import Image, ImageOps
import numpy as np
import cv2

def load_cn_model(model_dir):
  folder = model_dir
  file_name = 'diffusion_pytorch_model.safetensors'
  url = "https://huggingface.co/kataragi/ControlNet-LineartXL/resolve/main/Katarag_lineartXL-fp16.safetensors"

  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def load_cn_config(model_dir):
  folder = model_dir
  file_name = 'config.json'
  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
     config_path = os.path.join(os.getcwd(), file_name)
     shutil.copy(config_path, file_path)



def resize_image_aspect_ratio(image):
    # 元の画像サイズを取得
    original_width, original_height = image.size

    # アスペクト比を計算
    aspect_ratio = original_width / original_height

    # 標準のアスペクト比サイズを定義
    sizes = {
        1: (1024, 1024),  # 正方形
        4/3: (1152, 896),  # 横長画像
        3/2: (1216, 832),
        16/9: (1344, 768),
        21/9: (1568, 672),
        3/1: (1728, 576),
        1/4: (512, 2048),  # 縦長画像
        1/3: (576, 1728),
        9/16: (768, 1344),
        2/3: (832, 1216),
        3/4: (896, 1152)
    }

    # 最も近いアスペクト比を見つける
    closest_aspect_ratio = min(sizes.keys(), key=lambda x: abs(x - aspect_ratio))
    target_width, target_height = sizes[closest_aspect_ratio]

    # リサイズ処理
    resized_image = image.resize((target_width, target_height), Image.ANTIALIAS)

    return resized_image


def base_generation(size, color):
    canvas = Image.new("RGBA", size, color) 
    return canvas     