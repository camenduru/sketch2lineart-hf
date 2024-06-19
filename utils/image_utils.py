from PIL import Image, ImageOps
import numpy as np
import cv2

def canny_process(image_path, threshold1, threshold2):
    # 画像を開き、RGBA形式に変換して透過情報を保持
    img = Image.open(image_path)
    img = img.convert("RGBA")

    canvas_image = Image.new('RGBA', img.size, (255, 255, 255, 255))
    
    # 画像をキャンバスにペーストし、透過部分が白色になるように設定
    canvas_image.paste(img, (0, 0), img)

    # RGBAからRGBに変換し、透過部分を白色にする
    image_pil = canvas_image.convert("RGB")
    image_np = np.array(image_pil)
    
    # グレースケール変換
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Cannyエッジ検出
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    canny = Image.fromarray(edges)
    
    
    return canny


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