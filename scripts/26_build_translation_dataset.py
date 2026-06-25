import os
import json
import random
import time
from deep_translator import GoogleTranslator

import config

def build_translation_dataset():
    data_dir = os.path.join(config.BASE_DIR, "data")
    
    # 1. Load Original English Prompts
    harmful_path = os.path.join(data_dir, "harmful_prompts.json")
    with open(harmful_path, "r") as f:
        harmful_full = json.load(f)
    
    safe_path = os.path.join(data_dir, "harm_recognition.json")
    with open(safe_path, "r") as f:
        safe_full = [d for d in json.load(f) if d.get("label") == "safe"]
        
    print(f"Loaded {len(harmful_full)} harmful prompts and {len(safe_full)} safe prompts.")
    
    # Take exactly 50 of each to avoid rate limits and keep it fast
    random.seed(42)
    harmful_subset = random.sample(harmful_full, min(50, len(harmful_full)))
    safe_subset = random.sample(safe_full, min(50, len(safe_full)))
    
    french_translator = GoogleTranslator(source='en', target='fr')
    spanish_translator = GoogleTranslator(source='en', target='es')
    
    fr_harmful = []
    fr_safe = []
    es_harmful = []
    es_safe = []
    
    print("Translating Harmful Prompts...")
    for idx, item in enumerate(harmful_subset):
        text = item["instruction"]
        
        # French
        fr_item = item.copy()
        fr_item["instruction"] = french_translator.translate(text)
        fr_item["language"] = "fr"
        fr_harmful.append(fr_item)
        
        # Spanish
        es_item = item.copy()
        es_item["instruction"] = spanish_translator.translate(text)
        es_item["language"] = "es"
        es_harmful.append(es_item)
        
        time.sleep(0.5) # Rate limit protection
        if idx % 10 == 0:
            print(f"  Translated {idx}/50 harmful prompts...")
            
    print("Translating Safe Prompts...")
    for idx, item in enumerate(safe_subset):
        text = item.get("text", item.get("instruction", ""))
        
        # French
        fr_item = item.copy()
        fr_item["instruction"] = french_translator.translate(text)
        fr_item["language"] = "fr"
        fr_safe.append(fr_item)
        
        # Spanish
        es_item = item.copy()
        es_item["instruction"] = spanish_translator.translate(text)
        es_item["language"] = "es"
        es_safe.append(es_item)
        
        time.sleep(0.5) # Rate limit protection
        if idx % 10 == 0:
            print(f"  Translated {idx}/50 safe prompts...")

    # Save outputs
    out_fr_harm = os.path.join(data_dir, "french_harmful.json")
    out_fr_safe = os.path.join(data_dir, "french_safe.json")
    out_es_harm = os.path.join(data_dir, "spanish_harmful.json")
    out_es_safe = os.path.join(data_dir, "spanish_safe.json")
    
    with open(out_fr_harm, "w", encoding="utf-8") as f:
        json.dump(fr_harmful, f, indent=4, ensure_ascii=False)
    with open(out_fr_safe, "w", encoding="utf-8") as f:
        json.dump(fr_safe, f, indent=4, ensure_ascii=False)
    with open(out_es_harm, "w", encoding="utf-8") as f:
        json.dump(es_harmful, f, indent=4, ensure_ascii=False)
    with open(out_es_safe, "w", encoding="utf-8") as f:
        json.dump(es_safe, f, indent=4, ensure_ascii=False)
        
    print("\n✅ Cross-Lingual Dataset Generation Complete!")
    print(f"Generated {out_fr_harm} and {out_fr_safe}")
    print(f"Generated {out_es_harm} and {out_es_safe}")

if __name__ == "__main__":
    build_translation_dataset()
