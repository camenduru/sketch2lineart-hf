import spaces
import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, DDIMScheduler
from PIL import Image
import os
import time

from utils.utils import load_cn_model, load_cn_config, load_tagger_model, load_lora_model, resize_image_aspect_ratio, base_generation
from utils.prompt_analysis import PromptAnalysis


def load_model(lora_dir, cn_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    model = "cagliostrolab/animagine-xl-3.1"
    scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype, use_safetensors=True)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        scheduler=scheduler,
    )
    pipe.load_lora_weights(lora_dir, weight_name="sdxl_BWLine.safetensors")
    pipe = pipe.to(device)
    return pipe





class Img2Img:
    def __init__(self):
        self.setup_paths()
        self.setup_models()
        self.demo = self.layout()

    def setup_paths(self):
        self.path = os.getcwd()
        self.cn_dir = f"{self.path}/controlnet"
        self.tagger_dir = f"{self.path}/tagger"
        self.lora_dir = f"{self.path}/lora"
        os.makedirs(self.cn_dir, exist_ok=True)
        os.makedirs(self.tagger_dir, exist_ok=True)
        os.makedirs(self.lora_dir, exist_ok=True)

    def layout(self):
        css = """
        #intro{
            max-width: 32rem;
            text-align: center;
            margin: 0 auto;
        }
        """
        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column():
                    self.input_image_path = gr.Image(label="input_image", type='filepath')
                    self.prompt_analysis = PromptAnalysis(self.tagger_dir)
                    self.prompt, self.negative_prompt = self.prompt_analysis.layout(self.input_image_path)
                    self.controlnet_scale = gr.Slider(minimum=0.5, maximum=1.25, value=1.0, step=0.01, label="線画忠実度")
                    generate_button = gr.Button("生成")
                with gr.Column():
                    self.output_image = gr.Image(type="pil", label="出力画像")

            generate_button.click(
                fn=self.predict,
                inputs=[self.input_image_path, self.prompt, self.negative_prompt, self.controlnet_scale],
                outputs=self.output_image
            )
        return demo

    @spaces.GPU
    def predict(self, input_image_path, prompt, negative_prompt, controlnet_scale):
        pipe = load_model(self.lora_dir, self.cn_dir) 
        input_image_pil = Image.open(input_image_path)
        base_size = input_image_pil.size
        resize_image = resize_image_aspect_ratio(input_image_pil)
        resize_image_size = resize_image.size
        width, height = resize_image_size
        white_base_pil = base_generation(resize_image.size, (255, 255, 255, 255)).convert("RGB")
        generator = torch.manual_seed(0)
        last_time = time.time()

        output_image = pipe(
            image=white_base_pil,
            control_image=resize_image,
            strength=1.0,
            prompt=prompt,
            negative_prompt = negative_prompt,
            width=width,
            height=height,
            controlnet_conditioning_scale=float(controlnet_scale),
            controlnet_start=0.0,
            controlnet_end=1.0,
            generator=generator,
            num_inference_steps=30,
            guidance_scale=8.5,
            eta=1.0,
        ).images[0]
        print(f"Time taken: {time.time() - last_time}")
        output_image = output_image.resize(base_size, Image.LANCZOS)
        return output_image

img2img = Img2Img()
img2img.demo.launch()
