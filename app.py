import torch
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
from onediffx import compile_pipe
from io import BytesIO
import base64

class InferlessPythonModel:
    def initialize(self):
        repo_id = "SG161222/RealVisXL_V4.0_Lightning"
        
        self.pipe = DiffusionPipeline.from_pretrained( repo_id, torch_dtype=torch.float16,variant="fp16",
                                                      use_safetensors = True).to("cuda")
        self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
                                                                       use_karras_sigmas=True)
    
        print("COMPILE THE PIPELINE",flush=True)
        self.pipe = compile_pipe(self.pipe)
 

    def infer(self, inputs):
        prompt = inputs["prompt"]
        negative_prompt = inputs["negative_prompt"]
        image_output = self.pipe(prompt,
                                 negative_prompt = negative_prompt,
                                 num_inference_steps=5,
                                 guidance_scale=1).images[0]
         
        buff = BytesIO()
        image_output.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        
        return { "generated_image_base64" : img_str }

    def finalize(self):
        self.pipe = None 
