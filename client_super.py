import asyncio
from typing import Optional, Dict, List, Tuple
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
import json
import os
from openai import AsyncOpenAI

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.sessions: Dict[str, ClientSession] = {}  # 存储服务器名称到会话的映射
        self.exit_stack = AsyncExitStack()
        self.tools: Dict[str, List[Tool]] = {}  # 存储服务器名称到工具列表的映射
        self.tool_to_server: Dict[str, str] = {}  # 存储工具名称到服务器名称的映射
        
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
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        # 存储会话
        self.sessions[name] = session

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        print(f"{name}包含工具{response.tools}")
        
        # 存储工具列表
        self.tools[name] = response.tools
        
        # 记录每个工具属于哪个服务器
        for tool in response.tools:
            self.tool_to_server[tool.name] = name

    async def process_query(self, query: str) -> str:
        """使用 LLM 和所有连接的 MCP 服务器提供的工具处理查询
        
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

        # 整合所有服务器的工具
        all_tools = []
        for tool_list in self.tools.values():
            all_tools.extend(tool_list)
            
        # 转换为 OpenAI 工具格式
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in all_tools]

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

                # 查找工具所属的服务器
                server_name = self.tool_to_server.get(tool_name)
                if not server_name:
                    error_msg = f"找不到工具 {tool_name} 所属的服务器"
                    final_text.append(f"[错误: {error_msg}]")
                    continue
                
                # 获取对应服务器的会话
                session = self.sessions.get(server_name)
                if not session:
                    error_msg = f"找不到服务器 {server_name} 的会话"
                    final_text.append(f"[错误: {error_msg}]")
                    continue

                # 执行工具调用并记录结果
                try:
                    result = await session.call_tool(tool_name, tool_args)
                    final_text.append(f"[调用工具 {tool_name} (服务器: {server_name}), 参数: {tool_args}]")
                except Exception as e:
                    error_msg = f"调用工具 {tool_name} 失败: {str(e)}"
                    final_text.append(f"[错误: {error_msg}]")
                    result = type('obj', (object,), {'content': error_msg})

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
        # 给系统一点时间来关闭资源
        await asyncio.sleep(0.5)


async def main():
    """主函数，负责启动客户端并处理命令行参数"""
    client = MCPClient()
    try:
        # 从配置文件mcp.json中读取配置信息
        try:
            with open('mcp.json', 'r') as f:
                config = json.load(f)
            
            # 验证配置文件格式
            if 'mcpServers' not in config:
                print("错误：配置文件中缺少'mcpServers'键")
                return
            
            if not config['mcpServers']:
                print("警告：配置文件中没有定义服务器")
                return
                
            # 遍历配置文件中的服务器配置并连接
            for server_name, server_config in config['mcpServers'].items():
                # 验证服务器配置
                if 'command' not in server_config or 'args' not in server_config:
                    print(f"错误：服务器'{server_name}'配置不完整，跳过")
                    continue
                    
                await client.connect_to_server(
                    name=server_name,
                    command=server_config['command'],
                    args=server_config['args']
                )
                
            await client.chat_loop()
        except FileNotFoundError:
            print("错误：找不到mcp.json配置文件")
        except json.JSONDecodeError:
            print("错误：mcp.json不是有效的JSON文件")
    finally:
        await client.cleanup()
        # 确保所有任务都有机会完成
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
