"""V2 device optimization: keep spatial masks on the UNet execution device."""
class GPUInputPipeline:
    def __init__(self,pipeline):self.pipeline=pipeline
    def __getattr__(self,name):return getattr(self.pipeline,name)
    def __call__(self,*args,**kwargs):
        attention=kwargs.get('cross_attention_kwargs')
        if attention and 'ip_adapter_masks' in attention:
            attention=dict(attention)
            attention['ip_adapter_masks']=[mask.to(device=self.pipeline._execution_device) for mask in attention['ip_adapter_masks']]
            kwargs['cross_attention_kwargs']=attention
        return self.pipeline(*args,**kwargs)

def make_gpu_runtime():
    import torch
    from reference_inpaint import CACHE
    from diffusers import StableDiffusionXLInpaintPipeline,DDIMScheduler
    torch.set_num_threads(4)
    pipeline=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    pipeline.scheduler=DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_ip_adapter(str(CACHE/'reference_models'),subfolder='',weight_name='ip-adapter-faceid-portrait_sdxl.bin',image_encoder_folder=None,local_files_only=True)
    pipeline.enable_model_cpu_offload()
    return {'base_pipeline':GPUInputPipeline(pipeline)}
