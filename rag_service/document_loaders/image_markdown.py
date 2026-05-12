from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


class ImageMarkdown(BaseModel):
    markdown: str = ""
    object_key: str = ""
    url: str = ""


def upload_image_bytes(content: bytes, extension: str) -> ImageMarkdown:
    if not content:
        return ImageMarkdown()
    extension = _clean_extension(extension)
    try:
        from rag_service.utils.his_util.obs_util import upload_file_as_bytes

        object_key = upload_file_as_bytes(str(uuid.uuid4()), content)
        url = _format_image_download_url(object_key, extension)
        return ImageMarkdown(markdown=f"![]({url})", object_key=object_key, url=url)
    except Exception as exc:
        LOGGER.warning("Failed to upload parsed image: %s", exc)
        return ImageMarkdown()


def upload_image_bytes_as_markdown(content: bytes, extension: str) -> str:
    return upload_image_bytes(content, extension).markdown


def upload_image_bytes_as_url(content: bytes, extension: str) -> str:
    return upload_image_bytes(content, extension).url


def image_extension_from_partname(partname: Any, default: str = "png") -> str:
    suffix = str(partname or "").rsplit(".", 1)
    if len(suffix) == 2 and suffix[-1]:
        return _clean_extension(suffix[-1])
    return default


def _format_image_download_url(download_key: str, extension: str) -> str:
    try:
        from rag_service.constants import IMAGE_DOWNLOAD_URL_GAMMA, IMAGE_DOWNLOAD_URL_PROD
        from rag_service.env import ENV, EnvEnum

        image_url = IMAGE_DOWNLOAD_URL_PROD if ENV == EnvEnum.PROD else IMAGE_DOWNLOAD_URL_GAMMA
        return image_url.format(download_key, "image", extension)
    except Exception as exc:
        LOGGER.warning("Failed to format parsed image URL, falling back to object key: %s", exc)
        return download_key


def _clean_extension(extension: Optional[str]) -> str:
    extension = str(extension or "png").strip().lower()
    if extension.startswith("."):
        extension = extension[1:]
    return extension or "png"
