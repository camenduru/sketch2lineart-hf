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


def line_process(image_path, sigma, gamma):
    def DoG_filter(image, kernel_size=0, sigma=1.0, k_sigma=2.0, gamma=1.5):
        g1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        g2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma * k_sigma)
        return g1 - gamma * g2

    def XDoG_filter(image, kernel_size=0, sigma=1.4, k_sigma=1.6, epsilon=0, phi=10, gamma=0.98):
        epsilon /= 255
        dog = DoG_filter(image, kernel_size, sigma, k_sigma, gamma)
        dog /= dog.max()
        e = 1 + np.tanh(phi * (dog - epsilon))
        e[e >= 1] = 1
        return (e * 255).astype('uint8')

    def binarize_image(image):
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized

    def process_XDoG(image, kernel_size=0, sigma=1.4, k_sigma=1.6, epsilon=0, phi=10, gamma=0.98):
        xdog_image = XDoG_filter(image, kernel_size, sigma, k_sigma, epsilon, phi, gamma)
        binarized_image = binarize_image(xdog_image)
        final_image_pil = Image.fromarray(binarized_image)
        return final_image_pil

    # 画像を開き、RGBA形式に変換して透過情報を保持
    img = Image.open(image_path)
    img = img.convert("RGBA")

    canvas_image = Image.new('RGBA', img.size, (255, 255, 255, 255))
    
    # 画像をキャンバスにペーストし、透過部分が白色になるように設定
    canvas_image.paste(img, (0, 0), img)

    # RGBAからRGBに変換し、透過部分を白色にする
    image_pil = canvas_image.convert("RGB")
    
    # OpenCVが扱える形式に変換
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    inv_Line = process_XDoG(image_gray, kernel_size=0, sigma=sigma, k_sigma=1.6, epsilon=0, phi=10, gamma=gamma)
    return inv_Line



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