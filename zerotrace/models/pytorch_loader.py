import os
import shutil
import tempfile
import zipfile
from typing import Dict, Any, Tuple, Optional

TORCH_OK = False
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None

try:
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModel, AutoTokenizer as _AutoTok
    torch = _torch
    AutoModelForCausalLM = _AutoModel
    AutoTokenizer = _AutoTok
    TORCH_OK = True
except Exception:
    TORCH_OK = False


def load_local_archive_or_weights(file_obj, filename: str) -> Dict[str, Any]:
    if not TORCH_OK:
        return {"success": False, "error": "PyTorch is required for local models. Install with: pip install torch transformers"}

    ext = os.path.splitext(filename)[1].lower()
    tmp_dir = tempfile.mkdtemp()

    try:
        file_path = os.path.join(tmp_dir, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file_obj, f)

        if ext == ".zip":
            extract_dir = os.path.join(tmp_dir, "extracted")
            os.makedirs(extract_dir)
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(extract_dir)

            if not os.path.exists(os.path.join(extract_dir, "config.json")):
                subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
                if subdirs:
                    extract_dir = os.path.join(extract_dir, subdirs[0])

            tok = AutoTokenizer.from_pretrained(extract_dir)
            mdl = AutoModelForCausalLM.from_pretrained(extract_dir, torch_dtype=torch.float32, device_map="auto")
            model_name = filename.replace(".zip", "")
            params = sum(p.numel() for p in mdl.parameters())
            return {
                "success": True,
                "model_name": model_name,
                "params_b": round(params / 1e9, 4),
                "model": mdl,
                "tokenizer": tok,
            }

        elif ext in [".pt", ".pth", ".bin", ".safetensors"]:
            sd = torch.load(file_path, map_location="cpu", weights_only=True)
            params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
            return {
                "success": True,
                "model_name": filename,
                "params_b": round(params / 1e9, 4),
                "model": sd,
                "tokenizer": None,
                "note": "State dict loaded. Tokenizer not available for raw weights.",
            }
        else:
            return {
                "success": False,
                "error": f"Unsupported format: {ext}. Use .zip (full HF model) or .pt/.pth/.safetensors",
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_huggingface_model(model_id: str) -> Dict[str, Any]:
    if not TORCH_OK:
        return {"success": False, "error": "PyTorch is required for HuggingFace models. Install with: pip install torch transformers"}

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        params = sum(p.numel() for p in mdl.parameters())
        return {
            "success": True,
            "model_name": model_id,
            "params_b": round(params / 1e9, 4),
            "model": mdl,
            "tokenizer": tok,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
