"""PlantUML renderers. Each returns diagram text; syntax is correct by construction."""

from .class_ import render_class_diagram
from .component import render_component_diagram
from .deployment import render_deployment_diagram

__all__ = ["render_class_diagram", "render_component_diagram", "render_deployment_diagram"]
