import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "./LainChat"
OFFLOAD_DIR = "./offload"

MODE_LAIN = "lain"
MODE_QWEN = "qwen"

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

print("[+] Cli start.")

def build_prompt(mode, user_input):
    if mode == MODE_QWEN:
        return (
            f"Instruction: {user_input}\n"
            "Response:"
        )

    if mode == MODE_LAIN:
        return (
            "You are Lain Iwakura.\n"
            "This is not roleplay and not fiction.\n"
            "You answer once, do not continue or make a dialogue.\n"
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

    last_end = matches[-1].end()
    return text[:last_end].strip()

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

current_mode = MODE_LAIN
current_model = lain_model

print("=== LainChat ready ===")
print("Commands: /mode qwen | /mode lain | quit")

while True:
    user_input = input("[+]> ").strip()

    if user_input.lower() in ("quit", "exit"):
        print("[-] Cli quit.")
        break

    if user_input.startswith("/mode"):
        _, mode = user_input.split(maxsplit=1)
        if mode == MODE_QWEN:
            current_mode = MODE_QWEN
            current_model = base_model
            print("[*] Switched to Qwen base model")
        elif mode == MODE_LAIN:
            current_mode = MODE_LAIN
            current_model = lain_model
            print("[*] Switched to Lain model")
        else:
            print("[!] Unknown mode")
        continue

    prompt = build_prompt(current_mode, user_input)
    response = chat(current_model, prompt)
    print(f"[~]> {response}\n")
