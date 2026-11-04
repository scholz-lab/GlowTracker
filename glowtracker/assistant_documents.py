"""Files the user attaches to a message to the AI assistant: text, PDF and Word documents go to the
model as their text, images as pictures. An attachment is a plain dict, so it can be saved in the
conversation's transcript and sent again when the chat is continued. No Kivy import.

{'name': 'protocol.pdf', 'kind': 'text', 'text': '...', 'note': 'only the first ... characters'}
{'name': 'plate.tif', 'kind': 'image', 'url': 'data:image/jpeg;base64,...', 'note': ''}
"""
from __future__ import annotations

import base64
import html
import io
import os
import re
import zipfile

from ox.types import ImagePart, TextPart

MAX_FILE_BYTES = 50_000_000
MAX_TEXT_CHARS = 100_000        # about 25 000 tokens; a whole book would not fit a chat
MAX_IMAGE_SIDE = 1568           # larger images are scaled down by the API anyway

IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp', '.gif')


class AttachmentError(Exception):
    """The file cannot be attached, worded for the user."""


def load(path: str) -> dict:
    """Read a file for sending. Raises AttachmentError with the reason if that is not possible."""
    name = os.path.basename(path)
    extension = os.path.splitext(name)[1].lower()
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise AttachmentError(f'{name} is larger than {MAX_FILE_BYTES // 1_000_000} MB.')
        if extension in IMAGE_TYPES:
            return {'name': name, 'kind': 'image', 'url': _image_url(path), 'note': ''}
        if extension == '.pdf':
            text = _pdf_text(path)
        elif extension == '.docx':
            text = _docx_text(path)
        else:
            text = _plain_text(path)
    except AttachmentError:
        raise
    except Exception as e:
        raise AttachmentError(f'Could not read {name}: {e}') from e
    if not text.strip():
        raise AttachmentError(f'{name} has no text to send'
                              + (' (a scanned PDF? Attach the pages as images).' if extension == '.pdf' else '.'))
    note = ''
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        note = f'only the first {MAX_TEXT_CHARS // 1000}k characters'
    return {'name': name, 'kind': 'text', 'text': text, 'note': note}


def _plain_text(path: str) -> str:
    with open(path, 'rb') as f:
        data = f.read()
    if b'\0' in data[:8192]:
        raise AttachmentError(f'{os.path.basename(path)} is not a text file. PDF, Word (.docx), text '
                              f'and image files can be attached; save tables as CSV.')
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return data.decode('latin-1')


def _pdf_text(path: str) -> str:
    from pypdf import PdfReader
    pages = PdfReader(path).pages
    return '\n\n'.join(page.extract_text() or '' for page in pages)


def _docx_text(path: str) -> str:
    with zipfile.ZipFile(path) as docx:
        xml = docx.read('word/document.xml').decode('utf-8')
    paragraphs = [''.join(re.findall(r'<w:t(?: [^>]*)?>([^<]*)</w:t>', part)) for part in xml.split('</w:p>')]
    return html.unescape('\n'.join(paragraphs))


def _image_url(path: str) -> str:
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        image.seek(0)
        if image.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
            # microscope images: stretch the brightness, as a viewer would
            pixels = np.asarray(image, dtype=np.float32)
            low, high = np.percentile(pixels, (0.1, 99.9))
            pixels = np.clip((pixels - low) / max(float(high - low), 1e-6), 0, 1) * 255
            image = Image.fromarray(pixels.astype(np.uint8))
        image = image.convert('RGB')
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        data = io.BytesIO()
        image.save(data, 'JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(data.getvalue()).decode('ascii')


def user_content(text: str, attachments: list[dict] | None) -> str | list:
    """A user message's content: the text with the documents in it, plus any images as parts."""
    attachments = attachments or []
    documents = [f'[Attached document: {a["name"]}'
                 + (f' ({a["note"]})' if a.get('note') else '') + f']\n{a["text"]}\n[End of {a["name"]}]'
                 for a in attachments if a['kind'] == 'text']
    images = [a for a in attachments if a['kind'] == 'image']
    names = [f'[Attached image: {a["name"]}]' for a in images]
    full = '\n\n'.join(documents + names + ([text] if text else []))
    if not images:
        return full
    return [TextPart(text=full)] + [ImagePart(image_url={'url': a['url']}) for a in images]
