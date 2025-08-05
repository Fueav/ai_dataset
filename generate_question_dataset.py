#!/usr/bin/env python3
"""
通过deepseek API生成Merlin Chain function calling问题数据集
"""
import asyncio
import argparse
from typing import List, Dict
from deepseek_api_client import DeepSeekAPIClient
from dataset_utils import DatasetUtils

class QuestionDatasetGenerator:
    """问题数据集生成器，生成用户问题和引导对话"""
    
    def __init__(self, api_key: str):
        self.api_client = DeepSeekAPIClient(api_key)
        self.utils = DatasetUtils()
    
    async def generate_batch(self, system_prompt: str, batch_num: int, batch_size: int = 10) -> List[Dict]:
        """生成一批对话数据"""
        print(f"正在生成第 {batch_num} 批数据...")
        
        # 精简的prompt变体，覆盖所有Merlin MCP工具
        prompt_variations = [
            # 交易查询类
            f"生成{batch_size}个问题，问题属于以下主题：查询交易详情、获取区块交易、最新交易场景",
            # 地址分析类  
            f"生成{batch_size}个问题: 问题属于以下主题：钱包余额、地址详情、资产价值、持仓分析场景",
            # 代币信息类
            f"生成{batch_size}个问题: 问题属于以下主题：代币价格、市场数据、价格变化、交易量场景",
            # 区块查询类
            f"生成{batch_size}个问题: 问题属于以下主题：区块详情、最新区块、区块内交易等场景",
            # 持有者分析类
            f"生成{batch_size}个问题: 问题属于以下主题：代币持有者、转账记录、资产分布场景",
            # 搜索发现类
            f"生成{batch_size}个问题: 问题属于以下主题：链上搜索、代币查找、地址搜索场景",
            # Native代币类
            f"生成{batch_size}个问题: 问题属于以下主题：BTC价格、原生代币信息、实时价格场景",
            # 综合查询类
            f"生成{batch_size}个问题: 问题属于以下主题：混合查询，涵盖交易、地址、代币场景"
        ]
        
        # 根据批次号选择不同的prompt变体，并添加通用指导原则
        base_prompt = prompt_variations[batch_num % len(prompt_variations)]
        user_prompt = f"{base_prompt}，并注意以下几点：\n1.引导用户的参数例子使用随机生成的值。\n2.用户询问代币相关问题时不需要向用户索取代币合约地址。"
        
        print(f"  使用prompt变体 {batch_num % len(prompt_variations) + 1}: {user_prompt[:50]}...")
        
        response = self.api_client.call_api(system_prompt, user_prompt)
        if not response:
            print(f"第 {batch_num} 批数据生成失败")
            return []
        
        # 打印大模型的完整输出
        print(f"\n{'='*50}")
        print(f"大模型原始输出 (批次 {batch_num}):")
        print(f"{'='*50}")
        print(response)
        print(f"{'='*50}\n")
        
        conversations = self.utils.extract_json_from_response(response)
        if not conversations:
            print(f"第 {batch_num} 批数据解析失败")
            print("响应内容:", response[:500] + "..." if len(response) > 500 else response)
            return []
        
        print(f"第 {batch_num} 批成功生成 {len(conversations)} 个对话")
        return conversations
    
    async def generate_dataset(self, prompt_file: str, total_conversations: int = 6000, batch_size: int = 50, output_file: str = "function_calling_dataset.json"):
        """生成完整的问题数据集"""
        # 读取系统提示词
        system_prompt = self.utils.read_file(prompt_file)
        if not system_prompt:
            print("无法读取prompt文件")
            return
        
        all_conversations = []
        total_batches = (total_conversations + batch_size - 1) // batch_size
        
        print(f"开始生成问题数据集，目标: {total_conversations} 个对话，分 {total_batches} 批生成")
        
        for batch_num in range(1, total_batches + 1):
            try:
                batch_conversations = await self.generate_batch(system_prompt, batch_num, batch_size)
                all_conversations.extend(batch_conversations)
                
                # 保存中间结果
                self.utils.write_json_file(batch_conversations, f"temp_question_batch_{batch_num}.json")
                
                print(f"已生成 {len(all_conversations)} 个对话")
                
                # 避免API限流，间隔一段时间
                if batch_num < total_batches:
                    await asyncio.sleep(2)
                
            except Exception as e:
                print(f"生成第 {batch_num} 批数据时出错: {e}")
                continue
        
        # 保存最终结果
        self.utils.write_json_file(all_conversations, output_file)
        
        print(f"问题数据集生成完成！")
        print(f"总计生成: {len(all_conversations)} 个对话")
        print(f"保存至: {output_file}")
        
        return all_conversations

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成Merlin Chain function calling问题数据集")
    parser.add_argument("--total_conversations", type=int, default=6000, help="目标对话数量 (默认: 6000)")
    parser.add_argument("--batch_size", type=int, default=50, help="每批生成的对话数量 (默认: 50)")
    parser.add_argument("--output_file", type=str, default="function_calling_dataset_v2.json", help="输出文件名 (默认: function_calling_dataset_v2.json)")
    parser.add_argument("--prompt_file", type=str, default="prompt.txt", help="提示词文件 (默认: prompt.txt)")
    parser.add_argument("--api_key", type=str, help="DeepSeek API Key (可选，未提供则使用默认值)")
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 使用你的API配置
    api_key = args.api_key or "0f8fb6e0-c7b3-43a1-93af-17d8bb9da64c"  # 你的API Key
    
    print("🚀 开始使用深度求索API生成问题数据集...")
    print(f"   API Key前缀: {api_key[:8]}...")
    print("📝 生成function calling问题和引导对话")
    print(f"   目标对话数量: {args.total_conversations}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   输出文件: {args.output_file}")
    
    generator = QuestionDatasetGenerator(api_key)
    
    try:
        # 生成问题数据集
        conversations = await generator.generate_dataset(
            prompt_file=args.prompt_file,
            total_conversations=args.total_conversations,
            batch_size=args.batch_size,
            output_file=args.output_file
        )
        
        if conversations:
            print(f"✅ 问题数据集生成成功，共 {len(conversations)} 个对话")
            print("💡 下一步运行 generate_complete_dataset.py 来生成完整的function calling数据集")
        else:
            print("❌ 问题数据集生成失败")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())