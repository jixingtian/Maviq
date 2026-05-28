

import base64
import json
import re
from io import BytesIO
from typing import Optional

from PIL import Image
from loguru import logger


def pil_to_b64(image: Image.Image) -> str:

    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_data_url(image: Image.Image) -> str:

    return f"data:image/png;base64,{pil_to_b64(image)}"


def b64_to_pil(image_b64: str) -> Optional[Image.Image]:

    if not image_b64:
        return None
    try:
        img_bytes = base64.b64decode(image_b64)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


def data_url_to_pil(url: str) -> Optional[Image.Image]:

    try:
        b64 = url.split(",", 1)[1]
        return b64_to_pil(b64)
    except (IndexError, Exception) as e:
        logger.warning(f"Failed to parse data URL: {e}")
        return None


def extract_json_from_text(raw: str) -> Optional[dict]:

                      
    try:
        return json.loads(raw)
    except Exception:
        pass
                                      
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def build_single_turn_message(text: str, image: Optional[Image.Image] = None) -> list:

    content = []
    if image is not None:
        b64 = pil_to_b64(image)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]
