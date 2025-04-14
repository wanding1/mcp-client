import asyncio
from contextlib import AsyncExitStack
import json
import os
import sys
from typing import Optional
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    """MCP客户端类，用于与MCP服务器通信并处理LLM请求"""

    def __init__(self):
        """初始化MCP客户端"""
        self.session: Optional[ClientSession] = None  # MCP会话对象，初始为空
        self.exit_stack = AsyncExitStack()  # 异步退出栈，用于管理资源
        # 初始化OpenAI客户端，使用OpenRouter API
        self.client = AsyncOpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取API密钥
        )

    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务器
        
        Args:
            server_script_path: 服务器脚本的路径或npx命令
        """
        # 判断是否为npx命令

        if server_script_path.endswith('.js'):
            command = "node"
            args = [server_script_path]
        elif server_script_path.endswith('.py'):
            command = "python"
            args = [server_script_path]
        else:
            raise ValueError("不支持的文件类型，仅支持 .py 和 .js 文件")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        # 建立标准输入输出传输通道
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport  # 分离读写流

        # 创建并初始化MCP会话
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # 获取并显示可用的工具列表
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

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
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
