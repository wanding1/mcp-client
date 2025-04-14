import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
from contextlib import AsyncExitStack
import json
import os
import sys
from typing import Optional
from openai import AsyncOpenAI

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        # 初始化OpenAI客户端，使用OpenRouter API
        self.client = AsyncOpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取API密钥
        )

    async def connect_to_server(self, name, command, args):
        """Connect to an MCP server
        """
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )
        print(f"开始连接{name}")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        print(f'已连接{name}')
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        print(f"{name}包含工具{response.tools}")
        self.tools.append(response.tools)

    async def process_query(self, query: str) -> str:
        """使用 LLM 和 MCP 服务器提供的工具处理查询
        
        Args:
            query: 用户输入的查询字符串
            
        Returns:
            str: 处理后的响应文本
        """
        # 构建初始消息
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # 获取可用的工具列表
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # 调用LLM API获取初始响应
        response = await self.client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=available_tools
        )

        final_text = []
        message = response.choices[0].message
        final_text.append(message.content or "")

        # 处理工具调用循环
        while message.tool_calls:
            # 处理每个工具调用
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 执行工具调用并记录结果
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # 更新消息历史
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args)
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                })

            # 将工具调用结果反馈给LLM
            response = await self.client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                tools=available_tools
            )

            message = response.choices[0].message
            if message.content:
                final_text.append(message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """运行交互式聊天循环，处理用户输入并返回响应"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """清理资源"""

        await self.exit_stack.aclose()


async def main():
    """主函数，负责启动客户端并处理命令行参数"""
    client = MCPClient()
    try:
        #从配置文件mcp.json中读取配置信息
        with open('mcp.json', 'r') as f:
            config = json.load(f)
        # 遍历配置文件中的服务器配置并连接
        for server_name, server_config in config['mcpServers'].items():
            await client.connect_to_server(
                name=server_name,
                command=server_config['command'],
                args=server_config['args']
            )
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
