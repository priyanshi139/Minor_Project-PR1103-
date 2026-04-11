import os, json, zipfile
import keras

MODELS_DIR = r"C:\Users\Priya\OneDrive\Desktop\AQI_Project\models"

def fix_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        return {k: fix_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_config(i) for i in obj]
    return obj

fixed = 0
for fname in os.listdir(MODELS_DIR):
    if not fname.endswith("_lstm_v2.keras"):
        continue
    path = os.path.join(MODELS_DIR, fname)
    tmp  = path + "_tmp"
    print(f"Fixing: {fname}")
    try:
        with zipfile.ZipFile(path, 'r') as zin:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "config.json":
                        cfg = json.loads(data.decode("utf-8"))
                        cfg = fix_config(cfg)
                        data = json.dumps(cfg).encode("utf-8")
                    zout.writestr(item, data)
        os.replace(tmp, path)
        fixed += 1
        print(f"  Done!")
    except Exception as e:
        print(f"  Error: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)

print(f"\nFixed {fixed} models!")
print("Now testing load...")
test = os.path.join(MODELS_DIR, "Agartala_lstm_v2.keras")
m = keras.models.load_model(test, compile=False)
print("Loaded OK!", m.name)
