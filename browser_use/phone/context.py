"""
Playwright browser on steroids.
"""

import asyncio
import base64
import gc
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field

from browser_use.phone.views import PhoneState

logger = logging.getLogger(__name__)


class PhoneContextWindowSize(BaseModel):
    """Window size configuration for browser context"""

    width: int
    height: int

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields to ensure compatibility with dictionary
        populate_by_name=True,
        from_attributes=True,
    )

    # Support dict-like behavior for compatibility
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class PhoneContextConfig(BaseModel):
    """
    Configuration for the PhoneContext.

    Default values:
        cookies_file: None
            Path to cookies file for persistence

            disable_security: False
                    Disable browser security features (dangerous, but cross-origin iframe support requires it)

        minimum_wait_page_load_time: 0.5
            Minimum time to wait before getting page state for LLM input

            wait_for_network_idle_page_load_time: 1.0
                    Time to wait for network requests to finish before getting page state.
                    Lower values may result in incomplete page loads.

        maximum_wait_page_load_time: 5.0
            Maximum time to wait for page load before proceeding anyway

        wait_between_actions: 1.0
            Time to wait between multiple per step actions

        browser_window_size: PhoneContextWindowSize(width=1280, height=1100)
            Default browser window size

        no_viewport: False
            Disable viewport

        save_recording_path: None
            Path to save video recordings

        save_downloads_path: None
            Path to save downloads to

        trace_path: None
            Path to save trace files. It will auto name the file with the TRACE_PATH/{context_id}.zip

        locale: None
            Specify user locale, for example en-GB, de-DE, etc. Locale will affect navigator.language value, Accept-Language request header value as well as number and date formatting rules. If not provided, defaults to the system default locale.

        user_agent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
            custom user agent to use.

        highlight_elements: True
            Highlight elements in the DOM on the screen

        viewport_expansion: 0
            Viewport expansion in pixels. This amount will increase the number of elements which are included in the state what the LLM will see. If set to -1, all elements will be included (this leads to high token usage). If set to 0, only the elements which are visible in the viewport will be included.

        allowed_domains: None
            List of allowed domains that can be accessed. If None, all domains are allowed.
            Example: ['example.com', 'api.example.com']

        include_dynamic_attributes: bool = True
            Include dynamic attributes in the CSS selector. If you want to reuse the css_selectors, it might be better to set this to False.

              http_credentials: None
      Dictionary with HTTP basic authentication credentials for corporate intranets (only supports one set of credentials for all URLs at the moment), e.g.
      {"username": "bill", "password": "pa55w0rd"}

        is_mobile: None
            Whether the meta viewport tag is taken into account and touch events are enabled.

        has_touch: None
            Whether to enable touch events in the browser.

        geolocation: None
            Geolocation to be used in the browser context. Example: {'latitude': 59.95, 'longitude': 30.31667}

        permissions: None
            Phone permissions to grant. Values might include: ['geolocation', 'notifications']

        timezone_id: None
            Changes the timezone of the browser. Example: 'Europe/Berlin'
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
        populate_by_name=True,
        from_attributes=True,
        validate_assignment=True,
        revalidate_instances="subclass-instances",
    )

    cookies_file: str | None = None
    minimum_wait_page_load_time: float = 0.25
    wait_for_network_idle_page_load_time: float = 0.5
    maximum_wait_page_load_time: float = 5
    wait_between_actions: float = 0.5

    disable_security: bool = (
        False  # disable_security=True is dangerous as any malicious URL visited could embed an iframe for the user's bank, and use their cookies to steal money
    )

    browser_window_size: PhoneContextWindowSize = Field(
        default_factory=lambda: PhoneContextWindowSize(width=1280, height=1100)
    )
    no_viewport: Optional[bool] = None

    save_recording_path: str | None = None
    save_downloads_path: str | None = None
    save_har_path: str | None = None
    trace_path: str | None = None
    locale: str | None = None
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36  (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
    )

    highlight_elements: bool = True
    viewport_expansion: int = 0
    allowed_domains: list[str] | None = None
    include_dynamic_attributes: bool = True
    http_credentials: dict[str, str] | None = None

    keep_alive: bool = Field(
        default=False, alias="_force_keep_context_alive"
    )  # used to be called _force_keep_context_alive
    is_mobile: bool | None = None
    has_touch: bool | None = None
    geolocation: dict | None = None
    permissions: list[str] | None = None
    timezone_id: str | None = None


@dataclass
class CachedStateClickableElementsHashes:
    """
    Clickable elements hashes for the last state
    """

    url: str
    hashes: set[str]


class PhoneSession:
    def __init__(
        self,
        cached_state: PhoneState | None = None,
    ):
        self.cached_state = cached_state

        self.cached_state_clickable_elements_hashes: (
            CachedStateClickableElementsHashes | None
        ) = None


@dataclass
class PhoneContextState:
    """
    State of the phone context
    """

    target_id: str | None = None  # CDP target ID


class PhoneContext:
    def __init__(
        self,
        phone: "Phone",
        config: PhoneContextConfig | None = None,
        state: Optional[PhoneContextState] = None,
    ):
        self.context_id = str(uuid.uuid4())

        self.config = config or PhoneContextConfig(
            **(phone.config.model_dump() if phone.config else {})
        )
        self.phone = phone

        self.state = state or PhoneContextState()

        # Initialize these as None - they'll be set up when needed
        self.session: PhoneSession | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        pass

    async def _initialize_session(self):
        """Initialize the browser session"""
        logger.debug(f"🌎  Initializing new browser context with id: {self.context_id}")
        return self.session

    async def get_state(self) -> PhoneState:
        """Get the current state of the browser context"""
        screen_shot = ""
        touchable_elements = []
        return PhoneState(screenshot=screen_shot, touchable_elements=touchable_elements)
