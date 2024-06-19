import spaces
import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, DDIMScheduler
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
import os
import time

from utils.utils import load_cn_model, load_cn_config, load_tagger_model, load_lora_model, resize_image_aspect_ratio, base_generation
from utils.prompt_analysis import PromptAnalysis

class Img2Img:
    def __init__(self):
        self.setup_paths()
        self.setup_models()
        self.compel = self.setup_compel()
        self.demo = self.layout()

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.model = "cagliostrolab/animagine-xl-3.1"
        self.scheduler = DDIMScheduler.from_pretrained(self.model, subfolder="scheduler")
        self.controlnet = ControlNetModel.from_pretrained(self.cn_dir, torch_dtype=self.dtype, use_safetensors=True)
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            self.model,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            use_safetensors=True,
            scheduler=self.scheduler,
        )
        self.pipe.load_lora_weights(self.lora_dir, weight_name="sdxl_BWLine.safetensors")
        self.pipe = self.pipe.to(self.device)

    def setup_compel(self):
        return Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

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
                    self.input_image_path = gr.Image(label="入力画像", type='filepath')
                    self.prompt_analysis = PromptAnalysis(self.tagger_dir)
                    self.prompt, self.negative_prompt = self.prompt_analysis.layout(self.input_image_path)
                    self.controlnet_scale = gr.Slider(minimum=0.5, maximum=1.25, value=1.0, step=0.01, label="線画忠実度")
                    generate_button = gr.Button("生成")
                with gr.Column():
                    self.output_image = gr.Image(type="pil", label="生成画像")

            generate_button.click(
                fn=self.predict,
                inputs=[self.input_image_path, self.prompt, self.negative_prompt, self.controlnet_scale],
                outputs=self.output_image
            )
        return demo

    @spaces.GPU
    def predict(self, input_image_path, prompt, negative_prompt, controlnet_scale):
        input_image_pil = Image.open(input_image_path)
        base_size = input_image_pil.size
        resize_image = resize_image_aspect_ratio(input_image_pil)
        resize_image_size = resize_image.size
        width, height = resize_image_size
        white_base_pil = base_generation(resize_image.size, (255, 255, 255, 255)).convert("RGB")
        conditioning, pooled = self.compel([prompt, negative_prompt])
        generator = torch.manual_seed(0)
        last_time = time.time()

        output_image = self.pipe(
            image=white_base_pil,
            control_image=resize_image,
            strength=1.0,
            prompt_embeds=conditioning[0:1],
            pooled_prompt_embeds=pooled[0:1],
            negative_prompt_embeds=conditioning[1:2],
            negative_pooled_prompt_embeds=pooled[1:2],
            width=width,
            height=height,
            controlnet_conditioning_scale=float(controlnet_scale),
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

img2img = Img2Img()
img2img.demo.launch()
