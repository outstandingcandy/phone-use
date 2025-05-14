from langchain_openai import ChatOpenAI
from browser_use import Agent, MobileAgent, Browser, BrowserConfig

# from browser_use.controller.service import get
from dotenv import load_dotenv

load_dotenv()

import asyncio
import os

os.environ["OPENAI_API_KEY"] = "sk"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:4000/v1"
llm = ChatOpenAI(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        # browser_binary_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
        headless=False,
    )
)


async def main():
    agent = Agent(
        # task="1. Open ebay."
        # "2. Change the language to English."
        # "3. Search for 'matchbox'."
        # "4. Click each product on the first page."
        # "5. Extract the the product name and price."
        # "Note: If hit a CAPTCHA/verification page, please send a email to tangjiee@amazon.com and wait for 1 minutes.",
        # task="Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.",
        # task="Compare the prices of the following products on eBay: 'iPhone 14', 'Samsung Galaxy S21', and 'Google Pixel 6'.",
        # task="打开https://www.smzdm.com/fenlei/shengxianshipin/, 搜索'牛奶', 并提取前5个产品的名称和价格。依次打开每个产品的链接，提取购买方法，优惠券链接和价格，并将结果输出为JSON格式。",
        task="请帮我在携程上预订一张从北京到上海的机票，出发时间是下周一，返回时间是下周五。请确保选择最便宜的航班，并提供航班号和价格。",
        llm=llm,
        browser=browser,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
