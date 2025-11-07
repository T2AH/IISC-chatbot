import os
from typing import Optional, List

import google.generativeai as genai


class Gemini:
    """Gemini client that discovers supported models and tries them in a safe order."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("No Gemini API key provided (pass api_key or set GEMINI_API_KEY).")
        genai.configure(api_key=self.api_key)

        # Build candidate list
        self.candidates: List[str] = []
        # Track which model eventually worked
        self.selected_model: Optional[str] = None
        env_m = os.environ.get("GEMINI_MODEL")
        if model_name:
            self.candidates.append(model_name)
        if env_m and env_m not in self.candidates:
            self.candidates.append(env_m)
        # Discover models from API
        try:
            discovered = []
            for m in genai.list_models():
                # m may be an object with attributes .name and .supported_generation_methods
                name = getattr(m, "name", None) or str(m)
                methods = set(getattr(m, "supported_generation_methods", []) or [])
                if not name:
                    continue
                if "generateContent" in methods or "generate_content" in methods:
                    discovered.append(name)
            # Prefer gemini-1.5* models first
            preferred = [n for n in discovered if "gemini-1.5" in n]
            others = [n for n in discovered if n not in preferred]
            ordered = preferred + others
            # Include both full and short names (strip prefix 'models/')
            for n in ordered:
                if n not in self.candidates:
                    self.candidates.append(n)
                if "/" in n:
                    short = n.split("/")[-1]
                    if short and short not in self.candidates:
                        self.candidates.append(short)
        except Exception:
            # Fall back to common names if discovery fails
            for mname in [
                "gemini-1.5-flash",
                "gemini-1.5-flash-001",
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash-8b",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "gemini-pro",
            ]:
                if mname not in self.candidates:
                    self.candidates.append(mname)

    def generate(self, prompt: str, **kwargs) -> str:
        last_exc = None
        def expand(n: str) -> List[str]:
            opts = []
            # try as-is
            opts.append(n)
            # add fully-qualified
            if not n.startswith("models/"):
                opts.append(f"models/{n}")
            # add -001 and -latest variants for common names
            base_variants = []
            if n.endswith("-flash") or n.endswith("-pro"):
                base = n
                base_variants = [f"{base}-001", f"{base}-latest"]
            for v in base_variants:
                if v not in opts:
                    opts.append(v)
                if not v.startswith("models/"):
                    opts.append(f"models/{v}")
            return opts

        for name in self.candidates:
            try:
                success = False
                last_err = None
                for trial in expand(name):
                    try:
                        model = genai.GenerativeModel(trial)
                        resp = model.generate_content(prompt, **kwargs)
                        self.selected_model = trial
                        success = True
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if not success:
                    raise last_err or RuntimeError("All trials failed")
                # Preferred path: resp.text
                # Preferred path: resp.text
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    return resp.text
                # Fallback textualization
                return str(resp)
            except Exception as e:
                last_exc = e
                continue
        raise RuntimeError(f"Gemini generation failed for all candidates {self.candidates}: {last_exc}")
