from langchain_openai import ChatOpenAI
from browser_use import MobileAgent

# from browser_use.controller.service import get
from dotenv import load_dotenv

load_dotenv()

import asyncio
import os

os.environ["OPENAI_API_KEY"] = "sk"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:4000/v1"
llm = ChatOpenAI(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
# llm = ChatOpenAI(model="anthropic.claude-3-7-sonnet-20250219-v1:0")

# os.environ["OPENAI_API_KEY"] = os.environ.get("ALIYUN_API_KEY", "sk")
# os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# llm = ChatOpenAI(model="qwen-vl-max")


async def main():
    agent = MobileAgent(
        # task="1. Open ebay."
        # "2. Change the language to English."
        # "3. Search for 'matchbox'."
        # "4. Click each product on the first page."
        # "5. Extract the the product name and price."
        # "Note: If hit a CAPTCHA/verification page, please send a email to tangjiee@amazon.com and wait for 1 minutes.",
        # task="Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.",
        # task="Compare the prices of the following products on eBay: 'iPhone 14', 'Samsung Galaxy S21', and 'Google Pixel 6'.",
        # task="打开https://www.smzdm.com/fenlei/shengxianshipin/, 搜索'牛奶', 并提取前5个产品的名称和价格。依次打开每个产品的链接，提取购买方法，优惠券链接和价格，并将结果输出为JSON格式。",
        task="打开国家博物馆公众号，预约参观时间，选择5月15日的时间段，填写姓名和身份证号，提交预约申请。",
        llm=llm,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
