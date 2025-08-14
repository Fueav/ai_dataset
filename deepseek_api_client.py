#!/usr/bin/env python3
"""
DeepSeek API客户端封装
提供与DeepSeek API的交互功能，支持function calling
"""
import json
import requests
import asyncio
from typing import Dict, List, Optional
from merlin_mcp_client import MerlinMCPClient

class DeepSeekAPIClient:
    """DeepSeek API客户端，支持function calling"""
    
    def __init__(self, api_key: str, model: str , base_url: str ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def call_api_with_tools(self, system_prompt: str, user_prompt: str, tools_schema: List[Dict] = None, mcp_client: MerlinMCPClient = None, max_retries: int = 3) -> str:
        """调用deepseek API，支持function calling"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        # 如果有可用工具，添加到请求中
        if tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"
        
        for attempt in range(max_retries):
            try:
                print(f"  尝试第 {attempt + 1} 次API调用（支持function calling）...")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                result = response.json()
                message = result["choices"][0]["message"]
                
                # 处理工具调用
                if message.get("tool_calls") and mcp_client:
                    print(f"  检测到 {len(message['tool_calls'])} 个工具调用")
                    tool_responses = []
                    
                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        
                        print(f"    调用工具: {tool_name}")
                        print(f"    参数: {tool_args}")
                        
                        # 从完整工具名中提取MCP工具名
                        if tool_name.startswith("mcp__merlin_mcp_tool__"):
                            mcp_tool_name = tool_name.replace("mcp__merlin_mcp_tool__", "")
                            print(f"    映射工具名: {tool_name} -> {mcp_tool_name}")
                            # 调用MCP工具
                            tool_result = await mcp_client.call_tool(mcp_tool_name, tool_args)
                        else:
                            tool_result = {"error": f"Unknown tool: {tool_name}"}
                        
                        tool_response = {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(tool_result, ensure_ascii=False) if tool_result else "工具调用失败"
                        }
                        tool_responses.append(tool_response)
                    
                    # 继续对话，包含工具调用结果
                    follow_up_payload = {
                        "model": self.model,
                        "messages": payload["messages"] + [message] + tool_responses,
                        "temperature": 0.7,
                        "max_tokens": 4096
                    }
                    
                    if tools_schema:
                        follow_up_payload["tools"] = tools_schema
                        follow_up_payload["tool_choice"] = "auto"
                    
                    follow_up_response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=follow_up_payload,
                        timeout=120
                    )
                    follow_up_response.raise_for_status()
                    follow_up_result = follow_up_response.json()
                    
                    final_response = follow_up_result["choices"][0]["message"]["content"]
                    print(f"    工具调用完成，最终响应长度: {len(final_response)} 字符")
                    return final_response
                else:
                    print(f"    无工具调用，直接响应长度: {len(message['content'])} 字符")
                    return message["content"]
                
            except requests.exceptions.Timeout as e:
                print(f"  第 {attempt + 1} 次超时: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 10} 秒后重试...")
                    await asyncio.sleep((attempt + 1) * 10)
                continue
            except requests.exceptions.RequestException as e:
                print(f"  第 {attempt + 1} 次请求失败: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 5} 秒后重试...")
                    await asyncio.sleep((attempt + 1) * 5)
                continue
            except Exception as e:
                print(f"  第 {attempt + 1} 次处理失败: {e}")
                break
        
        print(f"所有重试失败，放弃API调用")
        return None
    
    async def call_api_with_tools_detailed(self, messages: List[Dict], tools_schema: List[Dict] = None, mcp_client: MerlinMCPClient = None, max_retries: int = 3) -> Dict:
        """调用deepseek API，支持function calling，返回详细信息包括工具调用记录"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        # 如果有可用工具，添加到请求中
        if tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"
        
        # 用于记录详细信息的变量
        tool_calls_info = []
        tools_used_info = []
        
        for attempt in range(max_retries):
            try:
                print(f"  尝试第 {attempt + 1} 次API调用（详细模式）...")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                result = response.json()
                message = result["choices"][0]["message"]
                
                # 处理工具调用
                if message.get("tool_calls") and mcp_client:
                    print(f"  检测到 {len(message['tool_calls'])} 个工具调用")
                    tool_responses = []
                    
                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        
                        print(f"    调用工具: {tool_name}")
                        print(f"    参数: {tool_args}")
                        
                        # 记录工具调用信息
                        tool_call_record = {
                            "id": tool_call["id"],
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        }
                        tool_calls_info.append(tool_call_record)
                        
                        # 从完整工具名中提取MCP工具名
                        if tool_name.startswith("mcp__merlin_mcp_tool__"):
                            mcp_tool_name = tool_name.replace("mcp__merlin_mcp_tool__", "")
                            print(f"    映射工具名: {tool_name} -> {mcp_tool_name}")
                            # 调用MCP工具
                            tool_result = await mcp_client.call_tool(mcp_tool_name, tool_args)
                        else:
                            tool_result = {"error": f"Unknown tool: {tool_name}"}
                        
                        # 记录工具详细信息
                        tool_info = {
                            "name": tool_name,
                            "description": self._get_tool_description(tool_name, tools_schema),
                            "arguments": tool_args,
                            "result": tool_result
                        }
                        tools_used_info.append(tool_info)
                        
                        tool_response = {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(tool_result, ensure_ascii=False) if tool_result else "工具调用失败"
                        }
                        tool_responses.append(tool_response)
                    
                    # 继续对话，包含工具调用结果
                    follow_up_payload = {
                        "model": self.model,
                        "messages": payload["messages"] + [message] + tool_responses,
                        "temperature": 0.7,
                        "max_tokens": 4096
                    }
                    
                    if tools_schema:
                        follow_up_payload["tools"] = tools_schema
                        follow_up_payload["tool_choice"] = "auto"
                    
                    follow_up_response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=follow_up_payload,
                        timeout=120
                    )
                    follow_up_response.raise_for_status()
                    follow_up_result = follow_up_response.json()
                    
                    final_response = follow_up_result["choices"][0]["message"]["content"]
                    print(f"    工具调用完成，最终响应长度: {len(final_response)} 字符")
                    
                    return {
                        "final_response": final_response,
                        "tool_calls": tool_calls_info,
                        "tools_used": tools_used_info
                    }
                else:
                    print(f"    无工具调用，直接响应长度: {len(message['content'])} 字符")
                    return {
                        "final_response": message["content"],
                        "tool_calls": [],
                        "tools_used": []
                    }
                
            except requests.exceptions.Timeout as e:
                print(f"  第 {attempt + 1} 次超时: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 10} 秒后重试...")
                    await asyncio.sleep((attempt + 1) * 10)
                continue
            except requests.exceptions.RequestException as e:
                print(f"  第 {attempt + 1} 次请求失败: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 5} 秒后重试...")
                    await asyncio.sleep((attempt + 1) * 5)
                continue
            except Exception as e:
                print(f"  第 {attempt + 1} 次处理失败: {e}")
                break
        
        print(f"所有重试失败，放弃API调用")
        return None
    
    def _get_tool_description(self, tool_name: str, tools_schema: List[Dict]) -> str:
        """从tools_schema中获取工具描述"""
        if not tools_schema:
            return ""
        
        for tool in tools_schema:
            if tool.get("function", {}).get("name") == tool_name:
                return tool.get("function", {}).get("description", "")
        
        return ""
    
    async def generate_complete_conversation(self, messages: List[Dict], tools_schema: List[Dict] = None, mcp_client: MerlinMCPClient = None, max_retries: int = 3, max_tool_rounds: int = 6) -> Dict:
        """生成完整的对话流程，包括function_call和observation记录，支持多轮工具调用"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        # 如果有可用工具，添加到请求中
        if tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"
        
        # 用于存储新消息的列表
        new_messages = []
        tools_used = []
        
        for attempt in range(max_retries):
            try:
                print(f"  尝试第 {attempt + 1} 次API调用（完整对话模式）...")
                
                # 初始化当前消息列表
                current_messages = payload["messages"].copy()
                tool_round = 0
                
                while tool_round < max_tool_rounds:
                    # 调用API
                    current_payload = {
                        "model": self.model,
                        "messages": current_messages,
                        "temperature": 0.7,
                        "max_tokens": 4096
                    }
                    
                    if tools_schema:
                        current_payload["tools"] = tools_schema
                        current_payload["tool_choice"] = "auto"
                    
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=current_payload,
                        timeout=120
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    print(f"API响应结果: {result}")

                    message = result["choices"][0]["message"]
                    
                    # 检查是否有工具调用
                    if message.get("tool_calls") and mcp_client:
                        tool_round += 1
                        print(f"  第 {tool_round} 轮工具调用，检测到 {len(message['tool_calls'])} 个工具调用")
                        tool_responses = []
                        
                        # 只执行第一个工具调用，让模型根据结果决定下一步
                        tool_call = message["tool_calls"][0]
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        
                        print(f"    执行第一个工具调用: {tool_name}")
                        print(f"    参数: {tool_args}")
                        if len(message["tool_calls"]) > 1:
                            print(f"    忽略其余 {len(message['tool_calls']) - 1} 个工具调用，等待模型根据第一个工具结果决定下一步")
                        
                        # 添加 function_call 消息
                        function_call_message = {
                            "from": "function_call",
                            "value": json.dumps({
                                "name": tool_name,
                                "arguments": tool_args
                            }, ensure_ascii=False)
                        }
                        new_messages.append(function_call_message)
                        
                        # 从完整工具名中提取MCP工具名
                        if tool_name.startswith("mcp__merlin_mcp_tool__"):
                            mcp_tool_name = tool_name.replace("mcp__merlin_mcp_tool__", "")
                            print(f"    映射工具名: {tool_name} -> {mcp_tool_name}")
                            # 调用MCP工具
                            tool_result = await mcp_client.call_tool(mcp_tool_name, tool_args)
                        else:
                            tool_result = {"error": f"Unknown tool: {tool_name}"}
                        
                        # 添加 observation 消息
                        observation_message = {
                            "from": "observation",
                            "value": json.dumps(tool_result, ensure_ascii=False) if tool_result else "工具调用失败"
                        }
                        new_messages.append(observation_message)
                        
                        # 记录工具信息
                        tool_info = {
                            "name": tool_name,
                            "description": self._get_tool_description(tool_name, tools_schema),
                            "parameters": self._get_tool_parameters(tool_name, tools_schema)
                        }
                        tools_used.append(tool_info)
                        
                        # 为API准备tool response（只包含第一个工具的结果）
                        tool_response = {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(tool_result, ensure_ascii=False) if tool_result else "工具调用失败"
                        }
                        tool_responses.append(tool_response)
                        
                        # 更新消息列表，准备下一轮调用
                        current_messages = current_messages + [message] + tool_responses
                        
                    else:
                        # 无工具调用，添加最终回复并结束循环
                        gpt_message = {
                            "from": "gpt",
                            "value": message["content"]
                        }
                        new_messages.append(gpt_message)
                        
                        if tool_round > 0:
                            print(f"    多轮工具调用完成（共 {tool_round} 轮），最终响应长度: {len(message['content'])} 字符")
                        else:
                            print(f"    无工具调用，直接响应长度: {len(message['content'])} 字符")
                        
                        return {
                            "new_messages": new_messages,
                            "tools_used_json": json.dumps(tools_used, ensure_ascii=False)
                        }
                
                # 如果达到最大工具调用轮数，强制结束
                print(f"  警告：达到最大工具调用轮数 {max_tool_rounds}，强制结束")
                if not new_messages or new_messages[-1].get("from") != "gpt":
                    # 如果没有最终回复，添加一个
                    gpt_message = {
                        "from": "gpt", 
                        "value": "由于达到最大工具调用轮数限制，对话被强制结束。"
                    }
                    new_messages.append(gpt_message)
                
                return {
                    "new_messages": new_messages,
                    "tools_used_json": json.dumps(tools_used, ensure_ascii=False)
                }
                
            except requests.exceptions.Timeout as e:
                print(f"  第 {attempt + 1} 次超时: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 10} 秒后重试...")
                    await asyncio.sleep((attempt + 1) * 10)
                continue
            except requests.exceptions.RequestException as e:
                print(f"  第 {attempt + 1} 次请求失败: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 5} 秒后重试...")
                    await asyncio.sleep((attempt + 1) * 5)
                continue
            except Exception as e:
                print(f"  第 {attempt + 1} 次处理失败: {e}")
                break
        
        print(f"所有重试失败，放弃API调用")
        return None
    
    def _get_tool_parameters(self, tool_name: str, tools_schema: List[Dict]) -> Dict:
        """从 tools_schema 中获取工具参数定义"""
        if not tools_schema:
            return {}
        
        for tool in tools_schema:
            if tool.get("function", {}).get("name") == tool_name:
                return tool.get("function", {}).get("parameters", {})
        
        return {}
    
    def call_api(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """调用deepseek API，不带function calling"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        for attempt in range(max_retries):
            try:
                print(f"  尝试第 {attempt + 1} 次API调用...")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.Timeout as e:
                print(f"  第 {attempt + 1} 次超时: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 10} 秒后重试...")
                    asyncio.sleep((attempt + 1) * 10)
                continue
            except requests.exceptions.RequestException as e:
                print(f"  第 {attempt + 1} 次请求失败: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 {(attempt + 1) * 5} 秒后重试...")
                    asyncio.sleep((attempt + 1) * 5)
                continue
            except Exception as e:
                print(f"  第 {attempt + 1} 次处理失败: {e}")
                break
        
        print(f"所有重试失败，放弃API调用")
        return None