import spaces
import gradio as gr
from gradio_imageslider import ImageSlider
import torch

torch.jit.script = lambda f: f
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    DDIMScheduler,
)
from controlnet_aux import AnylineDetector
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
import os
import time
import numpy as np

from utils.utils import load_cn_model, load_cn_config, load_tagger_model, load_lora_model, resize_image_aspect_ratio, base_generation
from utils.prompt_analysis import PromptAnalysis

path = os.getcwd()
cn_dir = f"{path}/controlnet"
os.makedirs(cn_dir)
tagger_dir = f"{path}/tagger"
os.mkdir(tagger_dir)
lora_dir = f"{path}/lora"
os.mkdir(lora_dir)

load_cn_model(cn_dir)
load_cn_config(cn_dir)
load_tagger_model(tagger_dir)
load_lora_model(lora_dir)

IS_SPACES_ZERO = os.environ.get("SPACES_ZERO_GPU", "0") == "1"
IS_SPACE = os.environ.get("SPACE_ID", None) is not None

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

LOW_MEMORY = os.getenv("LOW_MEMORY", "0") == "1"

print(f"device: {device}")
print(f"dtype: {dtype}")
print(f"low memory: {LOW_MEMORY}")


model = "cagliostrolab/animagine-xl-3.1"
scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    model,
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=True,
    scheduler=scheduler,
)

pipe.load_lora_weights(
    lora_dir, 
    weight_name="sdxl_BWLine.safetensors"
)


compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
)
pipe = pipe.to(device)



class Img2Img:
    def __init__(self):
        self.input_image_path = None

    @spaces.GPU
    def predict(
        self,
        input_image_path,
        prompt,
        negative_prompt,
        controlnet_conditioning_scale,
    ):
        input_image_pil = Image.open(input_image_path)
        base_size =input_image_pil.size
        resize_image= resize_image_aspect_ratio(input_image_pil)
        resize_image_size = resize_image.size
        width = resize_image_size[0]
        height = resize_image_size[1]
        white_base_pil = base_generation(resize_image.size, (255, 255, 255, 255)).convert("RGB")
        conditioning, pooled = compel([prompt, negative_prompt])
        generator = torch.manual_seed(0)
        last_time = time.time()

        output_image = pipe(
            image=white_base_pil,
            control_image=resize_image,
            strength=1.0,
            prompt_embeds=conditioning[0:1],
            pooled_prompt_embeds=pooled[0:1],
            negative_prompt_embeds=conditioning[1:2],
            negative_pooled_prompt_embeds=pooled[1:2],
            width=width,
            height=height,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            controlnet_start=0.0,
            controlnet_end=1.0,
            generator=generator,
            num_inference_steps=30,
            guidance_scale=8.5,
            eta=1.0,
        )
        print(f"Time taken: {time.time() - last_time}")
        output_image = output_image.resize(base_size, Image.LANCZOS)
        return output_image


    css = """
    #intro{
        # max-width: 32rem;
        # text-align: center;
        # margin: 0 auto;
    }
    """
    def layout(self,css):
        with gr.Blocks(css=css) as demo:
            with gr.Row() as block:
                with gr.Column():
                    # 画像アップロード用の行
                    with gr.Row():
                        with gr.Column():
                            self.input_image_path = gr.Image(label="入力画像",  type='filepath')
                    
                    # プロンプト入力用の行
                    with gr.Row():
                        prompt_analysis = PromptAnalysis(tagger_dir)
                        [prompt, nega] = prompt_analysis.layout(self.input_image_path)           
                    # 画像の詳細設定用のスライダー行
                    with gr.Row():
                        controlnet_conditioning_scale = gr.Slider(minimum=0.5, maximum=1.25, value=1.0, step=0.01, interactive=True, label="線画忠実度")
                
                    # 画像生成ボタンの行
                    with gr.Row():
                        generate_button = gr.Button("生成", interactive=False)

                with gr.Column():
                    output_image = gr.Image(type="pil", label="Output Image")

                # インプットとアウトプットの設定
                inputs = [
                    input_image_path,
                    prompt,
                    nega,
                    controlnet_conditioning_scale,
                ]
                outputs = [output_image]
                
                # ボタンのクリックイベントを設定
                generate_button.click(
                    fn=self.predict,
                    inputs=[self.input_image_path, prompt, nega, controlnet_conditioning_scale],
                    outputs=[output_image]
                )

        # デモの設定と起動
        demo.queue(api_open=True)
        demo.launch(show_api=True)