import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MAX_MEMORY_WORDS = 8192


torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

def chat(message:str, history, system_role:str, token:int=1024, temperature:float=0.9, top_p:float=0.9):
    if not message:
        return history, ""

    messages = []
    if system_role:
        messages.append({"role": "system", "content": system_role})

    memory_len = 0
    for h in history:
        req = h[0]
        messages.append({"role": "user", "content": req})
        memory_len += len(req)
        if memory_len >= MAX_MEMORY_WORDS:
            break

    messages.append({"role": "user", "content": message})

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": token,
        "return_full_text": False,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
    }
    
    output = pipe(messages, **generation_args)
    response = output[0]['generated_text']

    history.append((message, response))
    return history, ""

while True:
    print("You: ", end="", flush=True)
    input_text = None
    lines = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break
    input_text = "\n".join(lines)
    if not input_text:
        continue

    history, _ = chat(input_text, [], "", 128)
    print(history, flush=True)
