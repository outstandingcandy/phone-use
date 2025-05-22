import asyncio
import enum
import json
import logging
import re
from typing import Dict, Generic, Optional, Tuple, Type, TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from playwright.async_api import ElementHandle, Page

# from lmnr.sdk.laminar import Laminar
from pydantic import BaseModel

from browser_use.mobile_agent.views import PhoneActionModel, PhoneActionResult
from browser_use.phone.context import PhoneContext
from browser_use.phone_controller.registry.service import Registry
from browser_use.phone_controller.views import (
    DoneAction,
    InputTextAction,
    SearchWechatAction,
    TouchScreenAction,
    GetTouchableElementAction,
    NoParamsAction,
    SendKeysAction,
    ClickKeyboardAction,
)
from browser_use.utils import time_execution_sync
from browser_use.phone.utils import move_mouse, click_mouse, type_text

logger = logging.getLogger(__name__)


Context = TypeVar("Context")


class Controller(Generic[Context]):
    def __init__(
        self,
        exclude_actions: list[str] = [],
        output_model: Optional[Type[BaseModel]] = None,
    ):
        self.registry = Registry[Context](exclude_actions)

        """Register all default phone actions"""

        if output_model is not None:
            # Create a new model that extends the output model with success parameter
            class ExtendedOutputModel(BaseModel):  # type: ignore
                success: bool = True
                data: output_model  # type: ignore

            @self.registry.action(
                "Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached",
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return PhoneActionResult(
                    is_done=True,
                    success=params.success,
                    extracted_content=json.dumps(output_dict),
                )

        else:

            @self.registry.action(
                "Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached",
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return PhoneActionResult(
                    is_done=True, success=params.success, extracted_content=params.text
                )

        @self.registry.action(
            "Search the query in Wechat in the current tab, the query should be a search query like humans search in Wechat, concrete and not vague or super long. More the single most important items. "
            "You can use this to search for Wechat Public Account, Wechat articles, Wechat mini programs, etc.",
            param_model=SearchWechatAction,
        )
        async def search_wechat(params: SearchWechatAction):
            move_mouse(120, 150)
            click_mouse()
            click_mouse()
            type_text(params.query)
            type_text("\n")
            msg = f'🔍  Searched for "{params.query}" in Wechat'
            return PhoneActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Clicking the keyboard to type text - useful for mobile devices"
            "You can use this to type text into input fields, search boxes, etc.",
            param_model=ClickKeyboardAction,
        )
        async def click_keyboard(
            params: ClickKeyboardAction,
        ):
            for x, y in params.key_pos_list:
                move_mouse(x, y)
                click_mouse()
            msg = f"⌨️  Clicked keyboard at coordinates {params.key_pos_list}"
            logger.info(msg)
            return PhoneActionResult(extracted_content=msg, include_in_memory=True)

        # wait for x seconds
        @self.registry.action("Wait for x seconds default 3")
        async def wait(seconds: int = 3):
            msg = f"🕒  Waiting for {seconds} seconds"
            logger.info(msg)
            await asyncio.sleep(seconds)
            return PhoneActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Touch screen at coordinates - useful for mobile devices"
            "Change the coordinates percentage to absolute coordinates",
            param_model=TouchScreenAction,
        )
        async def touch_screen(params: TouchScreenAction):
            # computer_x = (params.left + params.right) / 2 * 330 + params.x_offset
            # computer_y = (params.top + params.bottom) / 2 * 750 + params.y_offset
            computer_x = (params.left + params.right) / 2 + params.x_offset
            computer_y = (params.top + params.bottom) / 2 + params.y_offset
            move_mouse(computer_x, computer_y)
            click_mouse()
            click_mouse()
            msg = f"🖱️  Touched screen at coordinates ({computer_x}, {computer_y})"
            logger.info(msg)
            return PhoneActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Send text to the screen - useful for mobile devices",
            param_model=InputTextAction,
        )
        async def send_text(
            params: InputTextAction,
        ):
            type_text(params.text)
            msg = f"⌨️  Sent text '{params.text}' to screen at coordinates ({params.x}, {params.y})"
            logger.info(msg)
            return PhoneActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Get all touchable elements on the screen - useful for mobile devices"
            "Before using touch screen action, you should use this action to get all touchable elements on the screen"
            "The output will be a list of objects, each object should contain the following information:"
            "1. The bounding box of the object in the form of [x1, y1, x2, y2]."
            "2. The label of the object."
            "3. The index of the object in the list.",
            param_model=GetTouchableElementAction,
        )
        async def get_touchable_elements(params: GetTouchableElementAction):
            message_contents = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + params.encoded_image
                    },
                },
                {
                    "type": "text",
                    "text": "Detect all objects in the image and output the bbox coordinates in JSON format."
                    "Here is the instruction:"
                    "The output should be a list of objects, each object should contain the following information:"
                    "1. The bounding box of the object in the form of [x1, y1, x2, y2]."
                    "2. The label of the object.",
                },
            ]
            import openai
            import re
            from openai.types.chat import ChatCompletionUserMessageParam

            client = openai.OpenAI(api_key="anything", base_url="http://127.0.0.1:4000")
            model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    ChatCompletionUserMessageParam(
                        role="user", content=message_contents
                    )
                ],
            )
            text_response = response.choices[0].message.content
            print(text_response)
            import re

            try:
                json_response = re.search(
                    r"```json(.*)```", text_response, re.DOTALL
                ).group(1)
            except AttributeError:
                return None
            print(json_response)
            bounding_boxes = json.loads(json_response)
            msg = f"🔍  Found {len(bounding_boxes)} touchable elements on the screen"
            logger.info(msg)
            return PhoneActionResult(extracted_content=msg, include_in_memory=True)

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    # Act --------------------------------------------------------------------

    @time_execution_sync("--act")
    async def act(
        self,
        action: PhoneActionModel,
        phone_context: PhoneContext,
        #
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[list[str]] = None,
        #
        context: Context | None = None,
    ) -> PhoneActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    # with Laminar.start_as_current_span(
                    #     name=action_name,
                    #     input={
                    #         'action': action_name,
                    #         'params': params,
                    #     },
                    #     span_type='TOOL',
                    # ):
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        phone=phone_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                        context=context,
                    )

                    # Laminar.set_span_output(result)

                    if isinstance(result, str):
                        return PhoneActionResult(extracted_content=result)
                    elif isinstance(result, PhoneActionResult):
                        return result
                    elif result is None:
                        return PhoneActionResult()
                    else:
                        raise ValueError(
                            f"Invalid action result type: {type(result)} of {result}"
                        )
            return PhoneActionResult()
        except Exception as e:
            raise e
