#!/usr/bin/env python3
"""
通过deepseek API生成Merlin Chain function calling问题数据集（智能去重版本）
基于去重管理器，避免重复生成，保持工具分布均衡
"""
import asyncio
import argparse
import signal
import atexit
from typing import List, Dict
from deepseek_api_client import DeepSeekAPIClient
from dataset_utils import DatasetUtils
from dedup_manager import DatasetDedupManager
from config_manager import ConfigManager


class SmartQuestionDatasetGenerator:
    """智能问题数据集生成器，具备去重和进度跟踪功能"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError("API Key未配置，请检查配置文件或环境变量")
        
        # 获取总对话数配置
        total_conversations = self.config.get('generation.default_total_conversations', 6000)
        
        self.api_client = DeepSeekAPIClient(api_key)
        self.utils = DatasetUtils()
        self.dedup_manager = DatasetDedupManager(total_conversations=total_conversations)
        self.output_file = self.config.get('generation.default_output_file', "function_calling_dataset_smart.json")
        self.current_conversations = []
        self._setup_cleanup_handlers()
    
    def _setup_cleanup_handlers(self):
        """设置清理处理器（类似Go的defer）"""
        # 注册程序正常退出时的清理
        atexit.register(self._emergency_cleanup)
        
        # 注册信号处理器处理Ctrl+C等中断
        signal.signal(signal.SIGINT, self._signal_cleanup_handler)
        signal.signal(signal.SIGTERM, self._signal_cleanup_handler)
        
        print("🛡️ 已设置应急清理机制（支持Ctrl+C安全退出）")
    
    def _signal_cleanup_handler(self, signum, frame):
        """信号处理器（处理Ctrl+C等中断信号）"""
        print(f"\n⚠️ 检测到中断信号 {signum}，正在安全退出...")
        self._emergency_cleanup()
        print("✅ 数据已安全保存，程序退出")
        exit(0)
    
    def _emergency_cleanup(self):
        """应急清理函数（类似defer）"""
        try:
            print("\n🚨 执行应急数据保护...")
            
            # 合并所有临时文件并保存
            if hasattr(self, 'current_conversations'):
                final_conversations = self._merge_all_temp_files(self.current_conversations)
                
                # 只保存到主输出文件（取消应急备份）
                self.utils.write_json_file(final_conversations, self.output_file)
                
                print(f"💾 数据已保存至: {self.output_file}")
                print(f"📊 共保存 {len(final_conversations)} 个对话")
                
        except Exception as e:
            print(f"❌ 应急清理失败: {e}")
    
    def _defer_cleanup(self):
        """手动触发清理（模拟defer调用）"""
        self._emergency_cleanup()
        

    async def generate_batch(self, base_system_prompt: str, batch_num: int, batch_size: int = 10) -> List[Dict]:
        """生成一批对话数据（集成去重逻辑）"""
        print(f"正在生成第 {batch_num} 批数据...")
        
        # 获取本批次的工具分配计划
        tool_allocation = self.dedup_manager._get_priority_tools(batch_size)
        if not tool_allocation:
            print(f"所有工具已完成目标，跳过批次 {batch_num}")
            return []
        
        # 检查是否启用并行生成
        enable_parallel = self.config.get('generation.enable_parallel_generation', True)
        max_concurrent = self.config.get('generation.max_concurrent_tools', 3)
        parallel_delay = self.config.get('generation.parallel_batch_delay', 0.5)
        
        batch_conversations = []
        
        if enable_parallel and len(tool_allocation) > 1:
            # 并行处理模式
            print(f"  🚀 启用并行处理，最大并发数: {max_concurrent}")
            
            # 使用信号量控制并发数量
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_tool_with_semaphore(tool_name: str, target_count: int):
                async with semaphore:
                    print(f"  🎯 [并行] 正在生成 {tool_name} 的 {target_count} 个问题...")
                    try:
                        result = await self._generate_for_specific_tool(
                            base_system_prompt, tool_name, target_count, batch_num
                        )
                        if parallel_delay > 0:
                            await asyncio.sleep(parallel_delay)
                        return tool_name, result
                    except Exception as e:
                        print(f"  ❌ [并行] 生成 {tool_name} 失败: {e}")
                        return tool_name, []
            
            # 创建并发任务
            tasks = []
            for tool_name, target_count in tool_allocation:
                if target_count > 0:
                    task = process_tool_with_semaphore(tool_name, target_count)
                    tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    print(f"  ❌ [并行] 任务执行异常: {result}")
                    continue
                
                tool_name, tool_conversations = result
                for conv_data in tool_conversations:
                    conversations_list = conv_data.get("conversations", [])
                    user_question = self._extract_user_question(conversations_list)
                    
                    if user_question and not self.dedup_manager.check_duplicate(user_question):
                        user_role = self._infer_user_role(user_question)
                        language_style = self._infer_language_style(user_question)
                        
                        self.dedup_manager.record_generated(
                            conversations_list, tool_name, user_role, language_style
                        )
                        
                        batch_conversations.append(conv_data)
                        print(f"    ✅ [并行] 记录: {tool_name} - {user_question[:50]}...")
        else:
            # 串行处理模式
            print(f"  📝 使用串行处理模式")
            for tool_name, target_count in tool_allocation:
                if target_count <= 0:
                    continue
                    
                print(f"  🎯 正在生成 {tool_name} 的 {target_count} 个问题...")
                
                # 为特定工具生成对话
                tool_conversations = await self._generate_for_specific_tool(
                    base_system_prompt, tool_name, target_count, batch_num
                )
                
                # 直接更新状态（不需要推断）
                for conv_data in tool_conversations:
                    conversations_list = conv_data.get("conversations", [])
                    user_question = self._extract_user_question(conversations_list)
                    
                    if user_question and not self.dedup_manager.check_duplicate(user_question):
                        # 直接使用已知的工具名更新状态
                        user_role = self._infer_user_role(user_question)
                        language_style = self._infer_language_style(user_question)
                        
                        self.dedup_manager.record_generated(
                            conversations_list, tool_name, user_role, language_style
                        )
                        
                        batch_conversations.append(conv_data)
                        print(f"    ✅ 记录: {tool_name} - {user_question[:50]}...")
        
        processing_mode = "并行" if enable_parallel and len(tool_allocation) > 1 else "串行"
        print(f"第 {batch_num} 批成功生成 {len(batch_conversations)} 个对话 ({processing_mode}处理)")
        return batch_conversations
    
    def _extract_user_question(self, conversations_list: List[Dict]) -> str:
        """提取用户问题"""
        for conv in conversations_list:
            if conv.get("from") == "user":
                return conv.get("value", "")
        return ""
    
    async def _generate_for_specific_tool(self, base_system_prompt: str, tool_name: str, 
                                        count: int, batch_num: int) -> List[Dict]:
        """为特定工具生成对话"""
        
        # 工具特定的生成提示
        tool_prompts = {
            "get_address_details_by_address": f"生成{count}个关于查询钱包地址详情、余额、基本信息的问题",
            "get_token_info_by_address": f"生成{count}个关于查询代币信息、代币详情的问题", 
            "list_address_latest_txs": f"生成{count}个关于查询地址最新交易记录、交易历史的问题",
            "get_tx_by_hash": f"生成{count}个关于通过交易哈希查询交易详情的问题",
            "search_chain_data": f"生成{count}个关于搜索链上数据、查找代币或地址的问题",
            "query_asset_value_by_address": f"生成{count}个关于查询地址总资产价值的问题",
            "query_token_holding_by_address": f"生成{count}个关于查询地址持仓分析、代币分布的问题",
            "get_block_by_number": f"生成{count}个关于通过区块号查询区块详情的问题",
            "list_latest_blocks": f"生成{count}个关于查询最新区块列表的问题",
            "get_token_priceChange_by_address": f"生成{count}个关于查询代币价格变化、涨跌幅的问题",
            "list_address_latest_token_transfers": f"生成{count}个关于查询代币转账记录的问题",
            "get_holders_by_address": f"生成{count}个关于查询代币持有者排行的问题",
            "batch_get_tx_by_hashes": f"生成{count}个关于批量查询多个交易哈希的问题",
            "list_block_txs": f"生成{count}个关于查询区块内交易列表的问题",
            "get_native_price_info_by_address": f"生成{count}个关于查询BTC原生代币价格的问题",
            "get_token_onChain_data_by_address": f"生成{count}个关于查询代币链上数据、交易量的问题",
            "list_recent_txs_num_by_address": f"生成{count}个关于查询地址交易数量统计的问题",
            "get_block_by_hash": f"生成{count}个关于通过区块哈希查询区块的问题",
            "list_latest_txs": f"生成{count}个关于查询最新交易列表的问题",
        }
        
        user_prompt = tool_prompts.get(tool_name, f"生成{count}个关于{tool_name}的问题")
        user_prompt += "，注意使用系统提示词里面给的示例地址和哈希参数，并且按照示例格式返回。"
        
        # 调用API生成
        response = self.api_client.call_api(base_system_prompt, user_prompt)
        if not response:
            return []
        
        # 解析并返回对话
        conversations = self.utils.extract_json_from_response(response)
        return conversations if conversations else []

    
    def _infer_user_role(self, question: str) -> str:
        """推断用户角色"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["怎么看", "成功没", "有没有", "是什么", "不懂"]):
            return "区块链小白"
        
        if any(word in question_lower for word in ["批量", "api", "接口", "调试", "erc-20"]):
            return "区块链开发者"
        
        if any(word in question_lower for word in ["持仓", "资产", "价格", "投资", "收益"]):
            return "区块链投资者"
        
        if any(word in question_lower for word in ["分析", "对比", "挖掘", "复杂"]):
            return "区块链专家"
        
        return "区块链小白"
    
    def _infer_language_style(self, question: str) -> str:
        """推断语言风格"""
        if any(word in question for word in ["ERC-20", "0x", "hash", "address", "token"]):
            return "技术用语"
        
        if any(word in question for word in ["address", "balance", "transaction"]) and \
           any('\u4e00' <= char <= '\u9fff' for char in question):
            return "中英混合"
        
        if "0x" in question and len([x for x in question.split() if x.startswith("0x") and len(x) < 42]):
            return "错误表达"
        
        return "口语化"

    def _merge_all_temp_files(self, current_conversations: List[Dict]) -> List[Dict]:
        """合并所有相关的临时文件"""
        import glob
        import os
        
        # 找到所有智能版本的临时文件
        temp_files = glob.glob("temp_smart_question_batch_*.json")
        
        all_conversations = []
        processed_files = []
        
        # 读取所有临时文件
        for temp_file in temp_files:
            try:
                temp_data = self.utils.load_json_file(temp_file)
                if temp_data:
                    all_conversations.extend(temp_data)
                    processed_files.append(temp_file)
            except Exception as e:
                print(f"    警告：读取临时文件 {temp_file} 失败: {e}")
        
        # 添加当前会话的对话
        all_conversations.extend(current_conversations)
        
        # 去重（基于conversations内容）
        unique_conversations = []
        seen_signatures = set()
        
        for conv_data in all_conversations:
            conversations_list = conv_data.get("conversations", [])
            if not conversations_list:
                continue
                
            # 提取用户问题作为唯一标识
            user_question = None
            for conv in conversations_list:
                if conv.get("from") == "user":
                    user_question = conv.get("value", "")
                    break
            
            if user_question:
                signature = self.dedup_manager.get_question_signature(user_question)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_conversations.append(conv_data)
        
        print(f"📁 合并了 {len(temp_files)} 个临时文件")
        print(f"📊 合并前总数: {len(all_conversations)}，去重后: {len(unique_conversations)}")
        
        # 合并完成后删除所有临时文件
        self._cleanup_temp_files(processed_files)
        
        return unique_conversations
    
    def _cleanup_temp_files(self, temp_files: List[str]):
        """清理临时文件（合并完成后删除所有临时文件）"""
        import os
        
        # 删除所有临时文件
        for file_path in temp_files:
            try:
                os.remove(file_path)
                print(f"    🗑️ 已删除临时文件: {file_path}")
            except Exception as e:
                print(f"    ⚠️ 删除文件失败 {file_path}: {e}")

    async def generate_dataset(self, prompt_file: str, total_conversations: int = 6000, batch_size: int = 50, output_file: str = "function_calling_dataset_smart.json"):
        """生成完整的问题数据集（智能去重版本）"""
        # 设置输出文件（用于应急清理）
        self.output_file = output_file
        
        # 读取系统提示词
        system_prompt = self.utils.read_file(prompt_file)
        if not system_prompt:
            print("无法读取prompt文件")
            return
        
        all_conversations = []
        total_batches = (total_conversations + batch_size - 1) // batch_size
        
        print(f"🧠 开始智能生成问题数据集，目标: {total_conversations} 个对话，分 {total_batches} 批生成")
        print(f"✨ 启用去重检查和进度跟踪")
        
        # 显示当前进度
        stats = self.dedup_manager.get_statistics()
        print(f"📊 当前进度: {stats['总进度']} ({stats['完成率']})")
        
        for batch_num in range(1, total_batches + 1):
            try:
                # 显示本批次目标
                batch_guidance = self.dedup_manager.get_next_batch_prompt(batch_size)
                priority_info = [line for line in batch_guidance.split('\n') if '分配' in line][:3]
                if priority_info:
                    print(f"🎯 第{batch_num}批次重点: {'; '.join(priority_info)}")
                
                batch_conversations = await self.generate_batch(system_prompt, batch_num, batch_size)
                all_conversations.extend(batch_conversations)
                
                # 更新当前对话状态（用于应急清理）
                self.current_conversations = all_conversations
                
                # 保存中间结果（添加时间戳避免覆盖）
                import time
                timestamp = int(time.time())
                self.utils.write_json_file(batch_conversations, f"temp_smart_question_batch_{batch_num}_{timestamp}.json")
                
                # 更新进度显示
                current_stats = self.dedup_manager.get_statistics()
                print(f"📈 累计生成: {len(all_conversations)} 个对话 (目标进度: {current_stats['完成率']})")
                
                # 每10批次显示详细统计
                if batch_num % 10 == 0:
                    print(f"\n📊 第{batch_num}批次完成，当前统计:")
                    for role, count in current_stats['角色分布'].items():
                        print(f"    {role}: {count}个")
                    print()
                
                # 避免API限流
                if batch_num < total_batches:
                    await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ 生成第 {batch_num} 批数据时出错: {e}")
                continue
        
        # 保存最终结果（合并所有临时文件）
        final_conversations = self._merge_all_temp_files(all_conversations)
        self.utils.write_json_file(final_conversations, output_file)
        
        # 正常完成，执行defer清理
        try:
            # 取消注册atexit，避免重复执行
            atexit.unregister(self._emergency_cleanup)
        except:
            pass
        
        # 生成最终报告
        final_stats = self.dedup_manager.get_statistics()
        print(f"\n🎉 智能问题数据集生成完成！")
        print(f"📈 总计生成: {len(final_conversations)} 个对话")
        print(f"💾 保存至: {output_file}")
        print(f"\n📊 最终统计:")
        print(f"   完成率: {final_stats['完成率']}")
        print(f"   已用问题模式: {final_stats['已用问题模式']}个")
        print(f"   已用地址: {final_stats['已用地址']}个")
        print(f"   已用交易哈希: {final_stats['已用交易哈希']}个")
        
        print(f"\n🔧 工具分布:")
        for tool, progress in list(final_stats['工具进度'].items())[:5]:
            print(f"   {tool}: {progress}")
        
        print(f"\n👥 用户角色分布:")
        for role, count in final_stats['角色分布'].items():
            print(f"   {role}: {count}个")
        
        print(f"🧹 正常完成，临时文件已清理")
        
        return final_conversations


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成Merlin Chain function calling问题数据集（智能去重版本）")
    parser.add_argument("--total_conversations", type=int, help="目标对话数量 (默认从配置文件读取)")
    parser.add_argument("--batch_size", type=int, help="每批生成的对话数量 (默认从配置文件读取)")
    parser.add_argument("--output_file", type=str, help="输出文件名 (默认从配置文件读取)")
    parser.add_argument("--prompt_file", type=str, help="提示词文件 (默认从配置文件读取)")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径 (默认: config.json)")
    parser.add_argument("--api_key", type=str, help="DeepSeek API Key (可选，会覆盖配置文件设置)")
    parser.add_argument("--reset", action="store_true", help="重置生成状态，从头开始")
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    print(f"📝 加载配置文件: {args.config}")
    config = ConfigManager(args.config)
    
    # 如果命令行提供了API Key，则覆盖配置文件
    if args.api_key:
        config.set_api_key(args.api_key)
        print("✅ 使用命令行提供的API Key")
    
    # 验证配置
    if not config.validate_config():
        print("❌ 配置验证失败，请检查配置文件")
        return
    
    # 重置状态选项
    if args.reset:
        import os
        try:
            os.remove("generation_state.json")
            print("🔄 已重置生成状态")
        except FileNotFoundError:
            pass
    
    # 从配置文件或命令行参数获取设置
    total_conversations = args.total_conversations or config.get('generation.default_total_conversations', 6000)
    batch_size = args.batch_size or config.get('generation.default_batch_size', 50)
    output_file = args.output_file or config.get('generation.default_output_file', "function_calling_dataset_smart.json")
    prompt_file = args.prompt_file or config.get('generation.default_prompt_file', "prompt.txt")
    
    api_key = config.get_api_key()
    print("🚀 开始使用深度求索API生成问题数据集（智能去重版本）...")
    print(f"   API Key前缀: {api_key[:8]}...")
    print("🧠 启用智能去重和进度跟踪功能")
    print(f"📝 生成function calling问题和引导对话")
    print(f"   目标对话数量: {total_conversations}")
    print(f"   批次大小: {batch_size}")
    print(f"   输出文件: {output_file}")
    print(f"   提示词文件: {prompt_file}")
    
    try:
        generator = SmartQuestionDatasetGenerator(config)
        
        # 生成问题数据集
        conversations = await generator.generate_dataset(
            prompt_file=prompt_file,
            total_conversations=total_conversations,
            batch_size=batch_size,
            output_file=output_file
        )
        
        if conversations:
            print(f"✅ 智能问题数据集生成成功，共 {len(conversations)} 个对话")
            print("💡 下一步运行 generate_complete_dataset.py 来生成完整的function calling数据集")
        else:
            print("❌ 问题数据集生成失败")
            
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        print("💡 请检查config.json文件或设置环境变量DEEPSEEK_API_KEY")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())