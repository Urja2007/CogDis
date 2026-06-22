import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

def test_model():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating...")
    outputs = model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Response: {response}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(base_dir, "model_load_success.txt")
    with open(out_file, "w") as f:
        f.write("Model loaded and generated successfully.\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Response: {response}\n")
        
    print(f"Saved success message to {out_file}")
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleanup complete.")

if __name__ == "__main__":
    test_model()
