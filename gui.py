import re
import tkinter as tk
import threading
from tkinter import scrolledtext, messagebox
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#tkinter made in some point with IA

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "./LainChat"
OFFLOAD_DIR = "./offload"

MODE_LAIN = "lain"
MODE_QWEN = "qwen"

BG = "#0f1117"
FG = "#e6e6e6"
ENTRY_BG = "#1a1c23"
BTN_BG = "#2a2d3a"
BTN_ACTIVE = "#3a3f55"
ACCENT = "#8ab4f8"

print("[*] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("[*] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", offload_folder=OFFLOAD_DIR, local_files_only=True, trust_remote_code=True)
base_model.config.pad_token_id = tokenizer.eos_token_id
base_model.eval()

print("[*] Loading Lain model...")
lain_model = PeftModel.from_pretrained(base_model, LORA_DIR, device_map="auto", local_files_only=True, offload_folder=OFFLOAD_DIR)
lain_model.eval()

print("[+] Gui start.")

current_mode = MODE_LAIN
current_model = lain_model

def build_prompt(mode, user_input):
    if mode == MODE_QWEN:
        return f"Instruction: {user_input}\nResponse:"

    if mode == MODE_LAIN:
        return (
            "You are Lain Iwakura.\n"
            "This is not roleplay and not fiction.\n"
            "You answer once, do not continue.\n"
            "You do not describe feelings, metaphors, or narratives.\n"
            "You are a quiet, highly skilled computer scientist.\n"
            "You answer concisely and technically.\n"
            "You do not ask questions.\n\n"
            f"Instruction: {user_input}\n"
            "Response:\n"
        )

    raise ValueError("Unknown mode")

def truncate_at_last_sentence(text):
    matches = list(re.finditer(r'[.!?]', text))
    if not matches:
        return text.strip()
    return text[:matches[-1].end()].strip()

ROLE_PATTERN = re.compile(
    r'(\n?\s*(###\s*)?(Human|Assistant|User|System)\s*[:#]\s*)+',
    re.IGNORECASE
)

def suppress_roles(text: str) -> str:
    text = ROLE_PATTERN.sub('', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chat(model, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3, top_p=0.85, repetition_penalty=1.15, no_repeat_ngram_size=4, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = decoded[len(prompt):]
    raw = suppress_roles(raw)
    raw = truncate_at_last_sentence(raw)
    return raw


def send_prompt():
    user_text = input_box.get("1.0", tk.END).strip()
    if not user_text:
        return

    input_box.delete("1.0", tk.END)
    response_box.insert(tk.END, f"\n[YOU]\n{user_text}\n")
    response_box.see(tk.END)

    thinking_index = response_box.index(tk.END)
    response_box.insert(tk.END, "[thinking....]\n")
    response_box.mark_set("thinking_start", "end-2l")
    response_box.see(tk.END)

    def run_model():
        try:
            prompt = build_prompt(current_mode, user_text)
            response = chat(current_model, prompt)

            def update_ui():
                response_box.delete("thinking_start", "thinking_start lineend +1c")

                response_box.insert(
                    thinking_index,
                    f"[{current_mode.upper()}]\n{response}\n"
                )
                response_box.see(tk.END)

            root.after(0, update_ui)

        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", str(e)))

    threading.Thread(target=run_model, daemon=True).start()


def set_lain_mode():
    global current_mode, current_model
    current_mode = MODE_LAIN
    current_model = lain_model
    status_label.config(text="Mode: Lain")

def set_qwen_mode():
    global current_mode, current_model
    current_mode = MODE_QWEN
    current_model = base_model
    status_label.config(text="Mode: Qwen")

def quit_app():
    print("[-] Gui quit.")
    root.destroy()

root = tk.Tk()
root.title("LainChat")
root.geometry("900x600")
root.configure(bg=BG)

response_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg=ENTRY_BG, fg=FG, insertbackground=FG, font=("JetBrains Mono", 11))
response_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

input_box = tk.Text(root, height=4, bg=ENTRY_BG, fg=FG, insertbackground=FG, font=("JetBrains Mono", 11))
input_box.pack(fill=tk.X, padx=10)

controls = tk.Frame(root, bg=BG)
controls.pack(fill=tk.X, padx=10, pady=5)

def dark_button(parent, text, command):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=BTN_BG,
        fg=FG,
        activebackground=BTN_ACTIVE,
        activeforeground=FG,
        relief=tk.FLAT,
        padx=10
    )

dark_button(controls, "Lain Mode", set_lain_mode).pack(side=tk.LEFT, padx=5)
dark_button(controls, "Qwen Mode", set_qwen_mode).pack(side=tk.LEFT)

status_label = tk.Label(controls, text="Mode: Lain", bg=BG, fg=FG)
status_label.pack(side=tk.LEFT, padx=20)

dark_button(controls, "Quit", quit_app).pack(side=tk.RIGHT, padx=5)

send_button = dark_button(controls, "Send", send_prompt)
send_button.pack(side=tk.RIGHT, padx=5)

root.bind("<Escape>", lambda e: quit_app())
root.bind("<Control-Return>", lambda e: send_prompt())

root.mainloop()
