import spaces
import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, DDIMScheduler
from PIL import Image
import os
import time

from utils.utils import load_cn_model, load_cn_config, load_tagger_model, load_lora_model, resize_image_aspect_ratio, base_generation
from utils.prompt_utils import remove_color
from utils.tagger import modelLoad, analysis


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
        self.post_filter = True
        self.tagger_model = None
        self.input_image_path = None

    def setup_paths(self):
        self.path = os.getcwd()
        self.cn_dir = f"{self.path}/controlnet"
        self.tagger_dir = f"{self.path}/tagger"
        self.lora_dir = f"{self.path}/lora"
        os.makedirs(self.cn_dir, exist_ok=True)
        os.makedirs(self.tagger_dir, exist_ok=True)
        os.makedirs(self.lora_dir, exist_ok=True)

    def setup_models(self):
        load_cn_model(self.cn_dir)
        load_cn_config(self.cn_dir)
        load_tagger_model(self.tagger_dir)
        load_lora_model(self.lora_dir)


    def process_prompt_analysis(self, input_image_path):
        if self.tagger_model is None:
            self.tagger_model = modelLoad(self.tagger_dir)
        tags = analysis(input_image_path, self.tagger_dir, self.tagger_model)
        tags_list = tags      
        if self.post_filter:
            tags_list = remove_color(tags)
        return tags_list


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
                    self.prompt = gr.Textbox(label="prompt", lines=3)
                    self.negative_prompt = gr.Textbox(label="negative_prompt", lines=3, value="lowres, error, extra digit, fewer digits, cropped, worst quality,low quality, normal quality, jpeg artifacts, blurry")
                    prompt_analysis_button = gr.Button("prompt解析")
                    self.controlnet_scale = gr.Slider(minimum=0.5, maximum=1.25, value=1.0, step=0.01, label="線画忠実度")
                    generate_button = gr.Button("生成")
                with gr.Column():
                    self.output_image = gr.Image(type="pil", label="出力画像")


            prompt_analysis_button.click(
                        self.process_prompt_analysis,
                        inputs=[self.input_image_path],
                        outputs=self.prompt
            )


            generate_button.click(
                fn=self.predict,
                inputs=[self.input_image_path, self.prompt, self.negative_prompt, self.controlnet_scale],
                outputs=self.output_image
            )
        return demo

    @spaces.GPU
    def predict(self, input_image_path, prompt, negative_prompt, controlnet_scale):
        # モデルのロードをここに移動
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
            negative_prompt=negative_prompt,
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
img2img.demo.launch(share=True)
