"""
Petition templates. Stub implementation.
"""

from typing import List, Optional


def get_template(template_id: str) -> str:
    """Get template content by ID. Stub."""
    # TODO: load from DB or files
    return ""


def list_templates(petition_type: Optional[str] = None) -> List[dict]:
    """List available templates. Stub."""
    # TODO: query templates
    return []
