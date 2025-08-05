#!/usr/bin/env python3
"""
Merlin Chain MCP 客户端
支持SSE连接和多工具调用
"""
import json
import asyncio
import aiohttp
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MCPTool:
    """MCP工具信息"""
    name: str
    description: str
    parameters: Dict[str, Any]


class MerlinMCPClient:
    """Merlin Chain MCP 客户端，支持SSE和多工具调用"""
    
    def __init__(self, base_url: str = "https://mcp.merlinchain.io"):
        self.base_url = base_url
        self.sse_url = f"{base_url}/sse"
        self.session: Optional[aiohttp.ClientSession] = None
        self.tools: Dict[str, MCPTool] = {}
        self.connected = False
        self.logger = logging.getLogger(__name__)
        self._sse_connection = None
        self.session_endpoint = None
        
    async def connect(self) -> bool:
        """连接到MCP服务器并建立SSE连接"""
        try:
            if not self.session:
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    keepalive_timeout=300
                )
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        'User-Agent': 'MerlinMCP-Client/1.0',
                        'Accept': 'text/event-stream',
                        'Cache-Control': 'no-cache'
                    }
                )
            
            # 建立SSE连接
            await self._establish_sse_connection()
            
            # 获取工具列表
            await self._fetch_tools()
            
            self.connected = True
            self.logger.info(f"成功连接到MCP服务器: {self.base_url}, 工具数量: {len(self.tools)}")
            return True
            
        except Exception as e:
            self.logger.error(f"连接MCP服务器失败: {e}")
            self.connected = False
            raise ConnectionError(f"无法连接到MCP服务器 {self.base_url}: {e}")
    
    async def _establish_sse_connection(self):
        """建立SSE连接并获取会话端点"""
        try:
            self.logger.info(f"建立SSE连接: {self.sse_url}")
            self._sse_connection = await self.session.get(
                self.sse_url,
                headers={
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            
            if self._sse_connection.status == 200:
                self.logger.info("SSE连接建立成功")
                # 读取SSE流获取会话端点
                await self._read_sse_events()
            else:
                self.logger.warning(f"SSE连接状态异常: {self._sse_connection.status}")
                
        except Exception as e:
            self.logger.warning(f"SSE连接失败，将使用HTTP调用: {e}")
    
    async def _read_sse_events(self):
        """读取SSE事件流获取会话信息"""
        try:
            async for line in self._sse_connection.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('event: endpoint'):
                    continue
                elif line.startswith('data: '):
                    endpoint_path = line[6:]  # 去掉 'data: ' 前缀
                    self.session_endpoint = f"{self.base_url}{endpoint_path}"
                    self.logger.info(f"获取到会话端点: {self.session_endpoint}")
                    break
                    
        except Exception as e:
            self.logger.error(f"读取SSE事件失败: {e}")
    
    async def disconnect(self):
        """断开连接"""
        if self._sse_connection:
            try:
                self._sse_connection.close()
                self._sse_connection = None
            except Exception as e:
                self.logger.warning(f"关闭SSE连接时出错: {e}")
        
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        self.logger.info("已断开MCP服务器连接")
    
    async def _fetch_tools(self, max_retries: int = 2):
        """通过MCP协议获取工具列表，支持自动重连"""
        if not self.session_endpoint:
            raise ConnectionError("未获取到会话端点")
        
        for attempt in range(max_retries + 1):
            try:
                # 使用MCP协议获取工具列表
                request_data = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                }
                
                async with self.session.post(
                    self.session_endpoint,
                    json=request_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    result = await response.json()
                    if attempt == 0:  # 只在第一次尝试时记录详细日志
                        self.logger.info(f"工具列表响应状态: {response.status}, 内容: {result}")
                    
                    # 检查是否是session过期相关的错误
                    if (response.status in [401, 403] or 
                        (response.status == 400 and 'error' in result and 
                         any(keyword in str(result['error']).lower() for keyword in 
                             ['session', 'expired', 'invalid', 'unauthorized']))):
                        
                        if attempt < max_retries:
                            self.logger.warning(f"获取工具列表时检测到session过期，尝试重新连接 (第{attempt+1}次)")
                            await self._establish_sse_connection()  # 重新建立SSE连接获取新的session
                            continue
                        else:
                            raise ConnectionError("Session过期且重连失败")
                    
                    if response.status in [200, 202]:  # 接受200和202状态码
                        if 'result' in result and 'tools' in result['result']:
                            self._parse_mcp_tools(result['result']['tools'])
                            self.logger.info(f"成功获取 {len(self.tools)} 个工具")
                            return
                        elif 'result' in result:
                            self.logger.error(f"工具列表响应中没有tools字段: {result['result']}")
                        else:
                            self.logger.error(f"无效的工具列表响应: {result}")
                    else:
                        if attempt < max_retries:
                            self.logger.warning(f"获取工具列表失败，状态码: {response.status}，尝试重新连接 (第{attempt+1}次)")
                            await self._establish_sse_connection()
                            continue
                        else:
                            self.logger.error(f"获取工具列表失败，状态码: {response.status}, 响应: {result}")
                            raise ConnectionError(f"获取工具列表失败，状态码: {response.status}")
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries:
                    self.logger.warning(f"获取工具列表网络错误，尝试重新连接 (第{attempt+1}次): {e}")
                    await self._establish_sse_connection()
                    continue
                else:
                    self.logger.error(f"获取工具列表网络错误: {e}")
                    raise ConnectionError(f"获取工具列表网络错误: {e}")
                    
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"获取工具列表异常，尝试重新连接 (第{attempt+1}次): {e}")
                    await self._establish_sse_connection()
                    continue
                else:
                    self.logger.error(f"获取工具列表异常: {e}")
                    raise ConnectionError(f"获取工具列表失败: {e}")
                    
        raise ConnectionError("获取工具列表经过多次重试后仍然失败")
    
    def _parse_mcp_tools(self, tools_list: List[Dict]):
        """解析MCP工具数据"""
        self.tools.clear()
        
        try:
            for tool_info in tools_list:
                try:
                    # MCP协议的工具格式
                    name = tool_info.get('name', '')
                    description = tool_info.get('description', '')
                    
                    # MCP工具的参数在inputSchema中
                    input_schema = tool_info.get('inputSchema', {})
                    
                    # 去掉mcp__merlin_mcp_tool__前缀（如果有）
                    if name.startswith('mcp__merlin_mcp_tool__'):
                        clean_name = name.replace('mcp__merlin_mcp_tool__', '')
                    else:
                        clean_name = name
                    
                    if clean_name:
                        tool = MCPTool(
                            name=clean_name,
                            description=description,
                            parameters=input_schema
                        )
                        self.tools[clean_name] = tool
                        self.logger.debug(f"添加工具: {clean_name}")
                        
                except Exception as e:
                    self.logger.warning(f"解析单个工具信息失败: {e}")
            
            self.logger.info(f"成功解析 {len(self.tools)} 个工具")
            
        except Exception as e:
            self.logger.error(f"解析工具数据失败: {e}")
            raise ValueError(f"解析工具数据失败: {e}")
    
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """通过MCP协议调用工具，支持自动重连"""
        if not self.connected:
            await self.connect()
        
        if not self.session_endpoint:
            raise ConnectionError("未获取到会话端点")
            
        if tool_name not in self.tools:
            available_tools = list(self.tools.keys())
            self.logger.error(f"未知工具: {tool_name}, 可用工具: {available_tools}")
            raise ValueError(f"未知工具: {tool_name}, 可用工具: {available_tools}")
        
        for attempt in range(max_retries + 1):
            try:
                # 使用MCP协议调用工具
                request_data = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,  # 直接使用工具名，不加前缀
                        "arguments": parameters
                    }
                }
                
                async with self.session.post(
                    self.session_endpoint,
                    json=request_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    result = await response.json()
                    if attempt == 0:  # 只在第一次尝试时记录详细日志
                        self.logger.info(f"工具调用响应状态: {response.status}, 内容: {result}")
                    
                    # 检查是否是session过期相关的错误
                    if (response.status in [401, 403] or 
                        (response.status == 400 and 'error' in result and 
                         any(keyword in str(result['error']).lower() for keyword in 
                             ['session', 'expired', 'invalid', 'unauthorized']))):
                        
                        if attempt < max_retries:
                            self.logger.warning(f"检测到session过期，尝试重新连接 (第{attempt+1}次)")
                            await self.reconnect()
                            continue
                        else:
                            self.logger.error("多次重连后仍然失败")
                            raise ConnectionError("Session过期且重连失败")
                    
                    if response.status in [200, 202]:  # 接受200和202状态码
                        if 'result' in result:
                            return {
                                "tool": tool_name,
                                "parameters": parameters,
                                "result": result['result'],
                                "status": "success"
                            }
                        elif 'error' in result:
                            # 检查错误是否是session相关
                            error_msg = str(result['error']).lower()
                            if any(keyword in error_msg for keyword in ['session', 'expired', 'invalid']):
                                if attempt < max_retries:
                                    self.logger.warning(f"检测到session相关错误，尝试重新连接 (第{attempt+1}次)")
                                    await self.reconnect()
                                    continue
                            
                            self.logger.error(f"工具调用错误: {result['error']}")
                            return {
                                "tool": tool_name,
                                "parameters": parameters,
                                "error": result['error'],
                                "status": "error"
                            }
                        else:
                            self.logger.error(f"无效的工具调用响应: {result}")
                            return {
                                "tool": tool_name,
                                "parameters": parameters,
                                "error": "无效的响应格式",
                                "status": "error"
                            }
                    else:
                        if attempt < max_retries:
                            self.logger.warning(f"工具调用失败，状态码: {response.status}，尝试重新连接 (第{attempt+1}次)")
                            await self.reconnect()
                            continue
                        else:
                            self.logger.error(f"工具调用失败，状态码: {response.status}, 响应: {result}")
                            raise ConnectionError(f"工具调用失败，状态码: {response.status}")
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries:
                    self.logger.warning(f"网络错误，尝试重新连接 (第{attempt+1}次): {e}")
                    await self.reconnect()
                    continue
                else:
                    self.logger.error(f"工具调用网络错误: {tool_name}, {e}")
                    raise TimeoutError(f"调用工具 {tool_name} 网络错误: {e}")
                    
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"工具调用异常，尝试重新连接 (第{attempt+1}次): {e}")
                    await self.reconnect()
                    continue
                else:
                    self.logger.error(f"调用工具 {tool_name} 失败: {e}")
                    raise RuntimeError(f"调用工具 {tool_name} 失败: {e}")
                    
        # 如果执行到这里，说明所有重试都失败了
        raise RuntimeError(f"调用工具 {tool_name} 经过 {max_retries} 次重试后仍然失败")
    
    async def call_multiple_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并行调用多个工具"""
        if not tool_calls:
            return []
        
        self.logger.info(f"并行调用 {len(tool_calls)} 个工具")
        
        # 创建并发任务
        tasks = []
        for call in tool_calls:
            tool_name = call.get('name', '')
            parameters = call.get('arguments', {})
            task = self.call_tool(tool_name, parameters)
            tasks.append(task)
        
        # 并发执行
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果和异常
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "error": f"工具调用异常: {str(result)}",
                        "tool_index": i
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"并行工具调用失败: {e}")
            raise RuntimeError(f"并行工具调用失败: {e}")
    
    
    async def reconnect(self) -> bool:
        """重新连接到MCP服务器"""
        self.logger.info("尝试重新连接MCP服务器...")
        await self.disconnect()
        return await self.connect()
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[MCPTool]:
        """获取工具详细信息"""
        return self.tools.get(tool_name)
    
    def get_all_tools_info(self) -> Dict[str, MCPTool]:
        """获取所有工具信息"""
        return self.tools.copy()
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.session:
                return False
                
            # 尝试多个健康检查端点
            endpoints = [
                f"{self.base_url}/health",
                f"{self.base_url}/api/health",
                f"{self.base_url}/status"
            ]
            
            for endpoint in endpoints:
                try:
                    async with self.session.get(
                        endpoint,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            return True
                except Exception:
                    continue
                    
            return False
                
        except Exception as e:
            self.logger.warning(f"健康检查失败: {e}")
            return False
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()


# 使用示例
async def main():
    """使用示例"""
    logging.basicConfig(level=logging.INFO)
    
    # 单工具调用示例
    async with MerlinMCPClient() as client:
        # 获取工具列表
        tools = client.get_available_tools()
        print(f"可用工具: {tools[:5]}...")  # 只显示前5个
        
        # 单个工具调用
        result = await client.call_tool("get_tx_by_hash", {
            "hash": "0x3b2060db2444eb4cfecfa8b3e44584040b4eb175b04b4a8a7ad37743c09e50dc"
        })
        print(f"单工具调用结果: {result}")
        
        # 多工具并行调用
        tool_calls = [
            {"name": "get_address_details_by_address", "arguments": {"address": "0x123..."}},
            {"name": "list_latest_blocks", "arguments": {"timestamp": "1704067200"}},
            {"name": "search_chain_data", "arguments": {"query": "MERL"}}
        ]
        
        multi_results = await client.call_multiple_tools(tool_calls)
        print(f"多工具调用结果数量: {len(multi_results)}")
        
        # 健康检查
        health = await client.health_check()
        print(f"服务健康状态: {health}")


if __name__ == "__main__":
    asyncio.run(main())