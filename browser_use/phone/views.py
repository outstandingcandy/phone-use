from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

from browser_use.dom.history_tree_processor.service import DOMHistoryElement
from browser_use.dom.views import DOMState


# Pydantic
class TabInfo(BaseModel):
    """Represents information about a browser tab"""

    page_id: int
    url: str
    title: str
    parent_page_id: Optional[int] = (
        None  # parent page that contains this popup or cross-origin iframe
    )


class GroupTabsAction(BaseModel):
    tab_ids: list[int]
    title: str
    color: Optional[str] = "blue"


class UngroupTabsAction(BaseModel):
    tab_ids: list[int]


@dataclass
class TouchableElement:
    tag_name: str
    x: int
    y: int


@dataclass
class PhoneState:
    screenshot: Optional[str] = None
    touchable_elements: list[TouchableElement] = field(default_factory=list)


@dataclass
class PhoneStateHistory:
    touchable_elements: list[TouchableElement] = field(default_factory=list)
    screenshot: Optional[str] = None

class BrowserError(Exception):
    """Base class for all browser errors"""


class URLNotAllowedError(BrowserError):
    """Error raised when a URL is not allowed"""
