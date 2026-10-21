"""
Framework detectors: each takes an ExtractContext and appends Interfaces and
DataTypes to ctx.ir. Detectors read the parsed AST but, like the core
extractor, only record names, signatures and interface literals.
"""

from .cli import detect_argparse, detect_click_typer
from .env import detect_env_vars
from .compose import detect_compose
from .fastapi import detect_fastapi
from .gradio import detect_gradio
from .messaging import detect_pika
from .models import detect_data_types

DEFAULT_DETECTORS = [
    detect_data_types,      # first, so other detectors can link TypeRefs to DataTypes
    detect_fastapi,
    detect_argparse,
    detect_click_typer,
    detect_gradio,
    detect_pika,
    detect_env_vars,
    detect_compose,
]

__all__ = ["DEFAULT_DETECTORS"]
