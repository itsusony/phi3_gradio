import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MAX_MEMORY_WORDS = 8192


torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

def chat(message, history, system_role, token, temperature, top_p):
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

with gr.Blocks() as webapp:
    with gr.Row():
        with gr.Column():
            system_role = gr.Textbox("", label="System Role", lines=5)
        with gr.Column():
            token = gr.Slider(128, 1024*100, 8192, step=128, label="Max New Token")
            top_p = gr.Slider(0, 1, 0.9, step=0.1, label="top_p")
            temperature = gr.Slider(0, 1, 0.6, step=0.1, label="temperature")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(value="", label="Enter your message")
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot, system_role, token, temperature, top_p], [chatbot, msg])
    clear.click(lambda: [], None, chatbot, queue=False)

webapp.launch(share=True)
