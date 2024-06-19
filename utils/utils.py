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
    url = "  https://huggingface.co/2vXpSwA7/iroiro-lora/resolve/main/test_controlnet2/CN-anytest_v3-50000_fp16.safetensors"
    file_path = os.path.join(folder, file_name)
    if not os.path.exists(file_path):
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {file_name}')
        else:
            print(f'Failed to download {file_name}')
    else:
        print(f'{file_name} already exists.')

def load_cn_config(model_dir):
  folder = model_dir
  file_name = 'config.json'
  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
     config_path = os.path.join(os.getcwd(), file_name)
     shutil.copy(config_path, file_path)

def load_tagger_model(model_dir):
    model_id = 'SmilingWolf/wd-swinv2-tagger-v3'
    files = [
        'config.json', 'model.onnx', 'selected_tags.csv', 'sw_jax_cv_config.json'
    ]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for file in files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            url = f'https://huggingface.co/{model_id}/resolve/main/{file}'
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f'Downloaded {file}')
            else:
                print(f'Failed to download {file}')
        else:
            print(f'{file} already exists.')


def load_lora_model(model_dir):
    file_name = 'sdxl_BW_bold_Line.safetensors'
    file_path = os.path.join(model_dir, file_name)
    if not os.path.exists(file_path):
        url = "https://huggingface.co/tori29umai/lineart/blob/main/sdxl_BW_bold_Line.safetensors"
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {file_name}')
        else:
            print(f'Failed to download {file_name}')
    else:
        print(f'{file_name} already exists.')

 
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
    resized_image = image.resize((target_width, target_height), Image.LANCZOS)

    return resized_image


def base_generation(size, color):
    canvas = Image.new("RGBA", size, color) 
    return canvas     