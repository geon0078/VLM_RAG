import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


model_path = "AIDC-AI/Ovis2-8B"

torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    cache_dir="./hf_cache",
    device_map="auto",
    low_cpu_mem_usage=True,
)

tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# single-image input
image_path = '/home/aisw/Project/UST-ETRI-2025/data/text2.jpg'
images = [Image.open(image_path)]
max_partition = 9
text = 'Describe the image. in korean'
query = f'<image>\n{text}'

# format conversation
prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)
input_ids = input_ids.unsqueeze(0).to(device=model.device)
attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
if pixel_values is not None:
    pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
pixel_values = [pixel_values]

# generate output
with torch.inference_mode():
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )
    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f'Output:\n{output}')
