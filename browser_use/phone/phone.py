"""
Playwright browser on steroids.
"""

import asyncio
import gc
import logging
import os
import socket
import subprocess
from typing import Literal

import psutil
import requests
from dotenv import load_dotenv
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import (
    Playwright,
    async_playwright,
)
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

load_dotenv()

from browser_use.phone.context import PhoneContext, PhoneContextConfig
from browser_use.utils import time_execution_async

logger = logging.getLogger(__name__)

IN_DOCKER = os.environ.get("IN_DOCKER", "false").lower()[0] in "ty1"

class PhoneConfig(BaseModel):
    r"""
    Configuration for the Phone.
	"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
        populate_by_name=True,
        from_attributes=True,
        validate_assignment=True,
        revalidate_instances="subclass-instances",
    )

    new_context_config: PhoneContextConfig = Field(
        default_factory=PhoneContextConfig
    )


# @singleton: TODO - think about id singleton makes sense here
# @dev By default this is a singleton, but you can create multiple instances if you need to.
class Phone:

    def __init__(
        self,
        config: PhoneConfig | None = None,
    ):
        logger.debug("🌎  Initializing new phone")
        self.config = config or PhoneConfig()

    async def new_context(
        self, config: PhoneContextConfig | None = None
    ) -> PhoneContext:
        """Create a phone context"""
        phone_config = self.config.model_dump() if self.config else {}
        context_config = config.model_dump() if config else {}
        merged_config = {**phone_config, **context_config}
        return PhoneContext(**merged_config)