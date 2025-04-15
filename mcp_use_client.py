import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


async def main():
    """Run the example using a configuration file."""
    # Load environment variables
    load_dotenv()

    client = MCPClient.from_config_file(
        os.path.join("mcp.json")
    )

    # Create LLM
    llm = ChatOpenAI(base_url=os.getenv("BASE_URL"),
                     api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取API密钥
                     model=os.getenv("MODEL")
                     )

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    # Run the query
    result = await agent.run(
        "访问web.gstarcad.com，在登录页面，输入账号15921762908密码123456，点击登录按钮",
        max_steps=30,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    # Run the appropriate example
    asyncio.run(main())
