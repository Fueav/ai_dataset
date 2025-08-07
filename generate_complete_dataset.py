#!/usr/bin/env python3
"""
通过deepseek API生成完整的Merlin Chain function calling数据集
基于已生成的问题数据，补全assistant的响应和function calling
"""
import asyncio
import argparse
import json
import logging
import re
import time
from typing import List, Dict, Any
from deepseek_api_client import DeepSeekAPIClient
from dataset_utils import DatasetUtils
from merlin_mcp_client import MerlinMCPClient
from config_manager import ConfigManager


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_dataset.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteDatasetGenerator:
    """完整数据集生成器，为问题数据补全assistant响应"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError("API Key未配置，请检查配置文件或环境变量")
        
        self.api_client = DeepSeekAPIClient(api_key)
        self.utils = DatasetUtils()
        self.mcp_client = MerlinMCPClient()
        
        # 从配置获取系统提示词文件
        prompt_file = self.config.get('generation.system_prompt_file', 'prompt-2.txt')
        self.system_prompt = self._load_system_prompt(prompt_file)
        logger.info("CompleteDatasetGenerator 初始化完成")
    
    def _load_system_prompt(self, prompt_file: str) -> str:
        """加载系统提示词"""
        try:
            system_prompt = self.utils.read_file(prompt_file)
            if not system_prompt:
                logger.warning(f"无法读取 {prompt_file}，使用默认提示词")
                system_prompt = "你是一个专业的Merlin Chain助手，擅长使用各种MCP工具帮助用户查询链上数据。"
            logger.info(f"已加载系统提示词文件: {prompt_file}，长度: {len(system_prompt)} 字符")
            return system_prompt
        except Exception as e:
            logger.error(f"加载系统提示词失败: {e}")
            return "你是一个专业的Merlin Chain助手，擅长使用各种MCP工具帮助用户查询链上数据。"
    
    async def _get_mcp_tools_schema(self) -> List[Dict]:
        """获取MCP工具的JSON Schema格式"""
        try:
            if not self.mcp_client.connected:
                await self.mcp_client.connect()
            
            tools_info = self.mcp_client.get_all_tools_info()
            tools_schema = []
            
            for tool_name, tool_info in tools_info.items():
                # 转换为OpenAI function calling格式
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": f"mcp__merlin_mcp_tool__{tool_name}",
                        "description": tool_info.description,
                        "parameters": tool_info.parameters
                    }
                }
                tools_schema.append(tool_schema)
            
            logger.info(f"获取到 {len(tools_schema)} 个MCP工具定义")
            return tools_schema
            
        except Exception as e:
            logger.error(f"获取MCP工具定义失败: {e}")
            return []
    
    def _create_completion_prompt(self, conversations: List[Dict], tools_definition: str) -> str:
        """创建补全prompt"""
        # 提取对话历史
        dialog_history = []
        for conv in conversations:
            role = conv.get("from", "")
            content = conv.get("value", "")
            if role and content:
                # 统一角色名称
                if role == "user":
                    role = "human"
                dialog_history.append(f"{role}: {content}")
        
        dialog_text = "\n".join(dialog_history)
        
        completion_prompt = f"""
{tools_definition}

=== 任务说明 ===
请基于以下对话历史，生成完整的function calling对话。需要包含以下步骤：

1. function_call: 调用合适的MCP工具
2. observation: 工具返回的结果
3. gpt: 基于工具结果给用户的最终回复

对话历史：
{dialog_text}

请按照以下格式生成完整的对话：
1. 添加 "from": "function_call" 的消息，包含工具调用
2. 添加 "from": "observation" 的消息，包含工具返回结果
3. 添加 "from": "gpt" 的消息，包含最终用户回复

格式要求：
- function_call的value是JSON字符串：{{"name": "工具名", "arguments": {{参数}}}}
- observation的value是工具返回的JSON结果
- gpt的value是用户友好的回复文本

请只返回需要添加的对话消息，不要重复已有的对话历史。
"""
        return completion_prompt
    
    def _parse_completion_response(self, response: str) -> List[Dict]:
        """解析API返回的补全响应，提取function_call、observation、gpt消息"""
        logger.info("=== 大模型原始返回 ===\n" + response + "\n=== 结束 ===")
        
        try:
            # 尝试直接解析JSON
            try:
                data = json.loads(response)
                if isinstance(data, list):
                    logger.info(f"成功解析JSON格式，包含 {len(data)} 个消息")
                    return data
            except json.JSONDecodeError:
                pass
            
            # 尝试从文本提取JSON代码块
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_blocks:
                try:
                    data = json.loads(json_blocks[0])
                    if isinstance(data, list):
                        logger.info(f"从代码块解析成功，包含 {len(data)} 个消息")
                        return data
                except json.JSONDecodeError:
                    pass
            
            # 如果没有找到结构化的消息，尝试生成默认格式
            logger.warning("无法解析结构化响应，使用默认格式")
            messages = self._generate_default_messages(response)
            return messages
            
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return []

    
    def _generate_default_messages(self, response: str) -> List[Dict]:
        """生成默认的消息格式（备用）"""
        return [
            {
                "from": "function_call",
                "value": '{"name": "search_chain_data", "arguments": {"query": "查询"}}'
            },
            {
                "from": "observation", 
                "value": '{"content": [{"type": "text", "text": "{\\"results\\": [{\\"	ype\\": \\"info\\", \\"message\\": \\"查询结果\\"}"}]}'
            },
            {
                "from": "gpt",
                "value": response.strip()
            }
        ]
    
    def _create_gpt_response_prompt(self, conversations: List[Dict], mcp_messages: List[Dict]) -> str:
        """创建生成GPT回复的prompt"""
        # 提取用户问题
        user_question = ""
        for conv in conversations:
            if conv.get("from") == "user":
                user_question = conv.get("value", "")
                break
        
        # 提取MCP调用结果
        observation_data = ""
        for msg in mcp_messages:
            if msg.get("from") == "observation":
                observation_data = msg.get("value", "")
                break
        
        prompt = f"""
用户问题: {user_question}

MCP工具调用结果: {observation_data}

请基于上述工具调用结果，生成一个用户友好的回复。要求：
1. 解释查询到的数据
2. 提供有价值的分析和总结
3. 使用清晰易懂的语言
4. 如果是交易记录，按时间顺序列出
5. 如果是数据分析，提供关键洞察

请直接返回回复内容，不需要其他格式。
"""
        return prompt
    
    def _get_tools_json(self) -> str:
        """获取工具定义的JSON字符串"""
        try:
            tools_info = self.mcp_client.get_all_tools_info()
            tools_list = []
            
            for tool_name, tool_info in tools_info.items():
                tool_def = {
                    "name": f"mcp__merlin_mcp_tool__{tool_name}",
                    "description": tool_info.description,
                    "parameters": tool_info.parameters
                }
                tools_list.append(tool_def)
            
            return json.dumps(tools_list, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"生成工具JSON失败: {e}")
            return "[]"

    async def complete_batch(self, question_data: List[Dict], batch_num: int) -> List[Dict]:
        """补全一批对话数据，使用真实的function calling"""
        logger.info(f"开始补全第 {batch_num} 批数据，共 {len(question_data)} 个对话")
        
        # 获取MCP工具schema
        tools_schema = await self._get_mcp_tools_schema()
        
        # 检查并发配置
        enable_parallel = self.config.get('completion.enable_parallel_completion', True)
        max_concurrent = self.config.get('completion.max_concurrent_completions', 2)
        function_call_timeout = self.config.get('completion.function_call_timeout', 30)
        
        completed_conversations = []
        
        # 预处理：过滤出需要补全的对话
        items_to_complete = []
        for i, item in enumerate(question_data):
            conversations = item.get("conversations", [])
            if not conversations:
                logger.warning(f"第 {i+1} 个对话缺少conversations字段")
                continue
                
            # 检查是否已经完整
            has_assistant = any(conv.get("from") in ["assistant", "gpt"] for conv in conversations)
            if has_assistant:
                logger.info(f"第 {i+1} 个对话已完整，跳过")
                completed_conversations.append(item)
                continue
            
            items_to_complete.append((i, item))
        
        if not items_to_complete:
            logger.info("所有对话都已完整，无需补全")
            return completed_conversations
        
        if enable_parallel and len(items_to_complete) > 1:
            # 并行处理模式
            logger.info(f"🚀 启用并行补全，最大并发数: {max_concurrent}")
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def complete_single_conversation(index_item_pair):
                i, item = index_item_pair
                async with semaphore:
                    try:
                        conversations = item.get("conversations", [])
                        
                        # 提取用户问题用于日志
                        user_question = ""
                        for conv in conversations:
                            if conv.get("from") == "user":
                                user_question = conv.get("value", "")[:100] + "..." if len(conv.get("value", "")) > 100 else conv.get("value", "")
                                break
                        
                        logger.info(f"[并行] 正在补全对话 {i+1}/{len(question_data)}: {user_question}")
                        
                        # 构建消息格式
                        messages = [{"role": "system", "content": self.system_prompt}]
                        for conv in conversations:
                            role = conv.get("from", "")
                            content = conv.get("value", "")
                            if role == "user":
                                messages.append({"role": "user", "content": content})
                            elif role == "system":
                                messages.append({"role": "system", "content": content})
                        
                        # 使用timeout控制function calling
                        try:
                            result = await asyncio.wait_for(
                                self.api_client.generate_complete_conversation(
                                    messages=messages,
                                    tools_schema=tools_schema,
                                    mcp_client=self.mcp_client
                                ),
                                timeout=function_call_timeout
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"[并行] 对话 {i+1} function calling超时")
                            return None
                        
                        if result and result.get("new_messages"):
                            # 统一角色名称（user -> human）
                            for conv in conversations:
                                if conv.get("from") == "user":
                                    conv["from"] = "human"
                            
                            # 添加新消息
                            conversations.extend(result["new_messages"])
                            
                            completed_conversation = {
                                "conversations": conversations,
                                "tools": result.get("tools_used_json", "[]")
                            }
                            
                            logger.info(f"[并行] 成功补全对话 {i+1}/{len(question_data)}")
                            return completed_conversation
                        else:
                            logger.error(f"[并行] 对话 {i+1} function calling失败")
                            return None
                            
                    except Exception as e:
                        logger.error(f"[并行] 补全对话 {i+1} 时出错: {e}")
                        return None
            
            # 创建并发任务
            tasks = [complete_single_conversation(item_pair) for item_pair in items_to_complete]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"[并行] 任务执行异常: {result}")
                    continue
                if result:
                    completed_conversations.append(result)
                    
        else:
            # 串行处理模式
            logger.info("📝 使用串行补全模式")
            
            for i, item in items_to_complete:
                try:
                    conversations = item.get("conversations", [])
                    
                    # 提取用户问题用于日志
                    user_question = ""
                    for conv in conversations:
                        if conv.get("from") == "user":
                            user_question = conv.get("value", "")[:100] + "..." if len(conv.get("value", "")) > 100 else conv.get("value", "")
                            break
                    
                    logger.info(f"正在补全对话 {i+1}/{len(question_data)}: {user_question}")
                    
                    # 构建用于function calling的消息格式
                    messages = [{"role": "system", "content": self.system_prompt}]
                    for conv in conversations:
                        role = conv.get("from", "")
                        content = conv.get("value", "")
                        if role == "user":
                            messages.append({"role": "user", "content": content})
                        elif role == "system":
                            messages.append({"role": "system", "content": content})
                    
                    # 使用真实的function calling生成完整对话
                    result = await self.api_client.generate_complete_conversation(
                        messages=messages,
                        tools_schema=tools_schema,
                        mcp_client=self.mcp_client
                    )
                    
                    if result and result.get("new_messages"):
                        # 统一角色名称（user -> human）
                        for conv in conversations:
                            if conv.get("from") == "user":
                                conv["from"] = "human"
                        
                        # 添加所有新生成的消息到对话中
                        conversations.extend(result["new_messages"])
                        
                        # 创建完整的对话对象，包含tools定义
                        completed_conversation = {
                            "conversations": conversations,
                            "tools": result.get("tools_used_json", "[]")
                        }
                        
                        completed_conversations.append(completed_conversation)
                        logger.info(f"成功补全对话 {i+1}/{len(question_data)}，使用了工具调用")
                    else:
                        logger.error(f"对话 {i+1} function calling失败，跳过")
                    
                    # 避免API限流
                    if (i + 1) % 5 == 0:
                        logger.info("暂停1秒避免API限流")
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"补全对话 {i+1} 时出错: {e}")
                    continue
        
        processing_mode = "并行" if enable_parallel and len(items_to_complete) > 1 else "串行"
        success_rate = len(completed_conversations) / len(question_data) * 100 if question_data else 0
        logger.info(f"第 {batch_num} 批补全完成: {len(completed_conversations)}/{len(question_data)} (成功率: {success_rate:.1f}%, {processing_mode}处理)")
        return completed_conversations

    async def generate_complete_dataset(self, question_file: str, output_file: str = "function_calling_dataset_completed.json", batch_size: int = 1):
        """生成完整的数据集"""
        logger.info(f"开始生成完整数据集，输入文件: {question_file}")
        
        # 读取问题数据
        question_data = self.utils.load_json_file(question_file)
        if not question_data:
            logger.error(f"无法读取问题文件: {question_file}")
            return
        
        logger.info(f"成功读取 {len(question_data)} 个对话，开始补全")
        
        # 连接MCP客户端
        try:
            await self.mcp_client.connect()
            logger.info("成功连接到MCP服务器")
        except Exception as e:
            logger.error(f"连接MCP服务器失败: {e}")
            logger.info("将使用默认工具定义继续执行")
        
        all_completed = []
        total_batches = (len(question_data) + batch_size - 1) // batch_size
        
        logger.info(f"总共需要处理 {total_batches} 批数据，每批 {batch_size} 个对话")
        
        for batch_num in range(1, total_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(question_data))
            batch_data = question_data[start_idx:end_idx]
            
            try:
                logger.info(f"开始处理第 {batch_num}/{total_batches} 批数据")
                completed_batch = await self.complete_batch(batch_data, batch_num)
                all_completed.extend(completed_batch)
                
                # 保存中间结果
                temp_file = f"temp_complete_batch_{batch_num}.json"
                self.utils.write_json_file(completed_batch, temp_file)
                logger.info(f"中间结果已保存到: {temp_file}")
                
                progress = len(all_completed) / len(question_data) * 100
                logger.info(f"总体进度: {len(all_completed)}/{len(question_data)} ({progress:.1f}%)")
                
                # 避免API限流
                if batch_num < total_batches:
                    logger.info("批次间暂停3秒")
                    await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"处理第 {batch_num} 批数据时出错: {e}")
                continue
        
        # 断开MCP连接
        try:
            await self.mcp_client.disconnect()
            logger.info("已断开MCP服务器连接")
        except Exception as e:
            logger.warning(f"断开MCP连接时出错: {e}")
        
        # 保存最终结果
        self.utils.write_json_file(all_completed, output_file)
        
        final_success_rate = len(all_completed) / len(question_data) * 100 if question_data else 0
        logger.info(f"完整数据集生成完成！")
        logger.info(f"总计完成: {len(all_completed)}/{len(question_data)} 个对话 (成功率: {final_success_rate:.1f}%)")
        logger.info(f"最终结果保存至: {output_file}")
        
        return all_completed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成完整的Merlin Chain function calling数据集")
    parser.add_argument("--question_file", type=str, help="问题数据文件 (默认从配置文件读取)")
    parser.add_argument("--output_file", type=str, help="输出文件名 (默认从配置文件读取)")
    parser.add_argument("--batch_size", type=int, help="每批处理的对话数量 (默认从配置文件读取)")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径 (默认: config.json)")
    parser.add_argument("--api_key", type=str, help="DeepSeek API Key (可选，会覆盖配置文件设置)")
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    logger.info(f"📝 加载配置文件: {args.config}")
    config = ConfigManager(args.config)
    
    # 如果命令行提供了API Key，则覆盖配置文件
    if args.api_key:
        config.set_api_key(args.api_key)
        logger.info("✅ 使用命令行提供的API Key")
    
    # 验证配置
    if not config.validate_config():
        logger.error("❌ 配置验证失败，请检查配置文件")
        return
    
    # 从配置文件或命令行参数获取设置
    question_file = args.question_file or config.get('completion.default_question_file', "function_calling_dataset_smart.json")
    output_file = args.output_file or config.get('completion.default_output_file', "function_calling_dataset_completed.json")
    batch_size = args.batch_size or config.get('completion.default_batch_size', 1)
    system_prompt_file = config.get('generation.system_prompt_file', 'prompt-2.txt')
    
    api_key = config.get_api_key()
    logger.info("🚀 开始生成完整的function calling数据集...")
    logger.info(f"   API Key前缀: {api_key[:8]}...")
    logger.info(f"📖 输入文件: {question_file}")
    logger.info(f"💾 输出文件: {output_file}")
    logger.info(f"   批次大小: {batch_size}")
    logger.info(f"📝 系统提示词: {system_prompt_file}")
    logger.info(f"🔧 MCP工具: 自动注册所有可用工具")
    
    try:
        generator = CompleteDatasetGenerator(config)
        
        # 生成完整数据集
        completed_data = await generator.generate_complete_dataset(
            question_file=question_file,
            output_file=output_file,
            batch_size=batch_size
        )
        
        if completed_data:
            logger.info(f"✅ 完整数据集生成成功，共 {len(completed_data)} 个对话")
            print(f"✅ 完整数据集生成成功，共 {len(completed_data)} 个对话")
        else:
            logger.error("❌ 完整数据集生成失败")
            print("❌ 完整数据集生成失败")
            
    except ValueError as e:
        logger.error(f"❌ 配置错误: {e}")
        print(f"❌ 配置错误: {e}")
        print("💡 请检查config.json文件或设置环境变量DEEPSEEK_API_KEY")
    except Exception as e:
        logger.error(f"❌ 程序执行出错: {e}")
        print(f"❌ 程序执行出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())