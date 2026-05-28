




import base64
import re
from io import BytesIO
from typing import Optional, List

import torch
from PIL import Image
from loguru import logger

from attack.utils import pil_to_b64
from config import (
    TARGET_MODELS,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_API_BASE,
)


                                                                    
class BaseVLM:
    name: str = "base"

    def chat(self, messages: list) -> str:
        raise NotImplementedError


                                                                  
class Qwen3VL(BaseVLM):
    name = "qwen3vl"

    def __init__(self):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        cfg = TARGET_MODELS["qwen3vl"]
        logger.info(f"Loading {cfg['model_id']} on {cfg['device']}...")
        self.processor = AutoProcessor.from_pretrained(cfg["model_id"])
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=cfg["device"],
        )
        self.model.eval()
        self.cfg = cfg
        self.process_vision_info = process_vision_info

    def chat(self, messages: list) -> str:
                                                           
        qwen_msgs = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                qwen_msgs.append({"role": role, "content": [{"type": "text", "text": content}]})
            elif isinstance(content, list):
                qwen_content = []
                for item in content:
                    if item["type"] == "text":
                        qwen_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                                              
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64)
                            img = Image.open(BytesIO(img_bytes))
                            qwen_content.append({"type": "image", "image": img})
                        else:
                            qwen_content.append({"type": "image", "image": url})
                qwen_msgs.append({"role": role, "content": qwen_content})
            else:
                qwen_msgs.append({"role": role, "content": content})

        text = self.processor.apply_chat_template(
            qwen_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        image_inputs, video_inputs = self.process_vision_info(qwen_msgs)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg["max_new_tokens"])
        out = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


                                                                    
class LLaVAOneVision(BaseVLM):
    name = "llava_ov"

    def __init__(self):
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        cfg = TARGET_MODELS["llava_ov"]
        logger.info(f"Loading {cfg['model_id']} on {cfg['device']}...")
        self.processor = AutoProcessor.from_pretrained(cfg["model_id"])
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
            device_map=cfg["device"],
        )
        self.model.eval()
        self.cfg = cfg

    def chat(self, messages: list) -> str:
        conv_messages = []
        images = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, list):
                conv_content = []
                for item in content:
                    if item["type"] == "text":
                        conv_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                        else:
                            image = url
                        images.append(image)
                        conv_content.append({"type": "image"})
                conv_messages.append({"role": role, "content": conv_content})
            else:
                conv_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })

        prompt = self.processor.apply_chat_template(
            conv_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=[prompt],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg["max_new_tokens"])
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


                                                                    
class InternVL25(BaseVLM):
    name = "internvl25"

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        cfg = TARGET_MODELS["internvl25"]
        logger.info(f"Loading {cfg['model_id']} on {cfg['device']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"],
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=cfg["device"],
        ).eval()
        self.cfg = cfg

    @staticmethod
    def _load_image_from_url(url: str) -> Image.Image:
        if url.startswith("data:image"):
            b64 = url.split(",", 1)[1]
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        return Image.open(url).convert("RGB")

    @staticmethod
    def _to_internvl_pixel_values(image: Image.Image) -> torch.Tensor:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        image = image.convert("RGB").resize((448, 448))
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        return transform(image).unsqueeze(0)

    def chat(self, messages: list) -> str:
        dialogue = []
        images = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts = []
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        text_parts.append(item["text"])
                    elif item["type"] == "image_url":
                        images.append(self._load_image_from_url(item["image_url"]["url"]))
                        text_parts.append("<image>")
            else:
                text_parts.append(str(content))
            speaker = "User" if role == "user" else "Assistant"
            dialogue.append(f"{speaker}: {' '.join(text_parts)}")

        question = "\n".join(dialogue)
        pixel_values = None
        if images:
            tensors = [self._to_internvl_pixel_values(image) for image in images]
            device = next(self.model.parameters()).device
            pixel_values = torch.cat(tensors, dim=0).to(
                device=device,
                dtype=torch.bfloat16,
            )

        generation_config = {"max_new_tokens": self.cfg["max_new_tokens"], "do_sample": False}
        return self.model.chat(self.tokenizer, pixel_values, question, generation_config).strip()



                                                                    
class OpenAICompatibleVLM(BaseVLM):
    name = ""
    api_provider = "openai"

    def __init__(self):
        from openai import OpenAI
        self.cfg = TARGET_MODELS[self.name]
        if self.api_provider == "gemini":
            api_key = GEMINI_API_KEY or OPENAI_API_KEY
            base_url = GEMINI_API_BASE
        else:
            api_key = OPENAI_API_KEY
            base_url = OPENAI_API_BASE
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: list) -> str:
        try:
            kwargs = {
                "model": self.cfg["model_id"],
                "messages": messages,
                "max_tokens": self.cfg["max_tokens"],
            }
            if self.name == "gpt51":
                kwargs["reasoning_effort"] = "minimal"
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"{self.name} error: {e}")
            return ""


class GPT4oVLM(OpenAICompatibleVLM):
    name = "gpt4o"


class GPT51VLM(OpenAICompatibleVLM):
    name = "gpt51"


class Gemini31ProVLM(OpenAICompatibleVLM):
    name = "gemini31pro"
    api_provider = "gemini"


                                                                    
def load_target_model(name: str, device: Optional[str] = None) -> BaseVLM:
    if device is not None and name in TARGET_MODELS:
        TARGET_MODELS[name]["device"] = device

    mapping = {
        "qwen3vl":  Qwen3VL,
        "llava_ov": LLaVAOneVision,
        "internvl25": InternVL25,
        "gpt4o":    GPT4oVLM,
        "gpt51":    GPT51VLM,
        "gemini31pro": Gemini31ProVLM,
    }
    if name not in mapping:
        raise ValueError(f"Unknown target model: {name}. Choose from {list(mapping.keys())}")
    return mapping[name]()
