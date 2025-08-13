#!/usr/bin/env python3
"""
简单好用的数据集去重管理器
避免批次生成时的问题重复
"""

import json
import hashlib
from typing import Dict, List, Set, Any
from collections import defaultdict, Counter
import re


class DatasetDedupManager:
    def __init__(self, config_file: str = "generation_state.json", total_conversations: int = 6000):
        self.config_file = config_file
        self.total_conversations = total_conversations
        self.state = self._load_state()
        
        # 工具分布比例配置（基于prompt.txt的原始6000分布）
        self.tool_ratios = {
            "get_address_details_by_address": 0.10,    # 600/6000
            "get_token_info_by_address": 0.08,        # 480/6000
            "list_address_latest_txs": 0.08,          # 480/6000
            "get_tx_by_hash": 0.07,                   # 420/6000
            "search_chain_data": 0.07,                # 420/6000
            "query_asset_value_by_address": 0.05,     # 300/6000
            "query_token_holding_by_address": 0.05,   # 300/6000
            "get_block_by_number": 0.05,              # 300/6000
            "list_latest_blocks": 0.05,               # 300/6000
            "get_token_priceChange_by_address": 0.05, # 300/6000
            "list_address_latest_token_transfers": 0.05, # 300/6000
            "get_holders_by_address": 0.05,           # 300/6000
            "batch_get_tx_by_hashes": 0.03,           # 180/6000
            "list_block_txs": 0.03,                   # 180/6000
            "get_native_price_info_by_address": 0.03, # 180/6000
            "get_token_onChain_data_by_address": 0.03, # 180/6000
            "list_recent_txs_num_by_address": 0.03,   # 180/6000
            "get_block_by_hash": 0.03,                # 180/6000
            "list_latest_txs": 0.04,                  # 240/6000
        }
        
        # 根据总对话数动态计算工具目标
        self.tool_targets = self._calculate_tool_targets()
        
        # 用户角色分布
        self.user_roles = {
            "区块链小白": 0.30,
            "区块链开发者": 0.30,
            "区块链投资者": 0.25,
            "区块链专家": 0.15
        }
        
        # 语言风格分布
        self.language_styles = {
            "技术用语": 0.25,
            "口语化": 0.35,
            "错误表达": 0.15,
            "中英混合": 0.25
        }

    def _load_state(self) -> Dict:
        """加载生成状态"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 转换回正确的数据结构
                state = {
                    "generated_count": defaultdict(int, data.get("generated_count", {})),
                    "used_questions": set(data.get("used_questions", [])),
                    "used_parameters": defaultdict(set),
                    "role_count": defaultdict(int, data.get("role_count", {})),
                    "style_count": defaultdict(int, data.get("style_count", {})),
                    "question_patterns": defaultdict(int, data.get("question_patterns", {})),
                    "total_generated": data.get("total_generated", 0)
                }
                # 转换used_parameters
                for key, values in data.get("used_parameters", {}).items():
                    state["used_parameters"][key] = set(values)
                return state
        except FileNotFoundError:
            return {
                "generated_count": defaultdict(int),
                "used_questions": set(),
                "used_parameters": defaultdict(set),
                "role_count": defaultdict(int),
                "style_count": defaultdict(int),
                "question_patterns": defaultdict(int),
                "total_generated": 0
            }
    
    def _save_state(self):
        """保存生成状态"""
        # 转换set为list以便JSON序列化
        state_copy = dict(self.state)
        state_copy["used_questions"] = list(self.state["used_questions"])
        state_copy["used_parameters"] = {k: list(v) for k, v in self.state["used_parameters"].items()}
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(state_copy, f, ensure_ascii=False, indent=2)

    def get_question_signature(self, question: str) -> str:
        """生成问题特征签名"""
        # 移除地址、哈希等参数，保留问题模式
        cleaned = re.sub(r'0x[a-fA-F0-9]+', '[ADDRESS]', question)
        cleaned = re.sub(r'\d+', '[NUMBER]', cleaned)
        cleaned = re.sub(r'[^\u4e00-\u9fa5\w\s]', '', cleaned)  # 保留中文、英文、数字、空格
        return hashlib.md5(cleaned.strip().encode()).hexdigest()[:12]

    def check_duplicate(self, question: str, tool_name: str = None) -> bool:
        """检查是否重复"""
        signature = self.get_question_signature(question)
        
        # 检查问题模式重复
        if signature in self.state["used_questions"]:
            return True
            
        # 检查工具使用是否超限
        if tool_name and self.state["generated_count"][tool_name] >= self.tool_targets.get(tool_name, 0):
            return True
            
        return False

    def get_next_batch_prompt(self, batch_size: int = 50) -> str:
        """生成下一批次的动态prompt"""
        
        # 分析当前进度
        progress_info = self._analyze_progress()
        
        # 确定本批次重点工具
        priority_tools = self._get_priority_tools(batch_size)
        
        # 确定避免的问题模式
        avoid_patterns = self._get_avoid_patterns()
        
        # 确定角色和风格分布
        role_style_guide = self._get_role_style_guide(batch_size)
        
        # 构建动态prompt
        dynamic_prompt = f"""
基于以下进度和约束生成本批次数据：

【当前进度】
- 总完成: {self.state['total_generated']}/{self.total_conversations} ({progress_info['completion_rate']:.1%})
- 本批次目标: {batch_size}条

【本批次重点工具】（优先生成以下工具的问题）
{self._format_priority_tools(priority_tools)}

【严格避免以下已用问题模式】
{avoid_patterns}

【本批次角色和风格分布】
{role_style_guide}

【参数使用指导】
- 地址参数：优先使用未使用过的样本地址
- 已使用地址数量: {len(self.state['used_parameters']['addresses'])}
- 已使用交易哈希数量: {len(self.state['used_parameters']['tx_hashes'])}

【特别要求】
1. 每个问题必须独特，不能是已生成问题的简单变形
2. 优先生成进度落后的工具问题
3. 保持用户角色分布平衡
4. 确保参数不重复使用
"""
        
        return dynamic_prompt

    def _analyze_progress(self) -> Dict:
        """分析当前生成进度"""
        completion_rate = self.state['total_generated'] / self.total_conversations
        
        return {
            "completion_rate": completion_rate,
            "lagging_tools": [tool for tool, target in self.tool_targets.items() 
                            if self.state['generated_count'][tool] < target * completion_rate * 0.8]
        }

    def _get_priority_tools(self, batch_size: int) -> List[str]:
        """获取本批次优先工具"""
        tool_priorities = []
        
        for tool, target in self.tool_targets.items():
            current = self.state['generated_count'][tool]
            remaining = max(0, target - current)
            if remaining > 0:
                priority = remaining / target  # 剩余比例作为优先级
                tool_priorities.append((tool, remaining, priority))
        
        # 按优先级排序
        tool_priorities.sort(key=lambda x: x[2], reverse=True)
        
        # 分配本批次数量
        batch_allocation = []
        allocated = 0
        
        for tool, remaining, priority in tool_priorities:
            if allocated >= batch_size:
                break
            alloc = min(remaining, max(1, int(batch_size * priority)))
            batch_allocation.append((tool, alloc))
            allocated += alloc
            
        return batch_allocation

    def _get_avoid_patterns(self) -> str:
        """获取需要避免的问题模式"""
        if len(self.state["used_questions"]) == 0:
            return "无（首批生成）"
        
        # 取最近的一些问题模式作为避免示例
        recent_patterns = list(self.state["used_questions"])[-20:]
        avoid_text = "避免生成与以下模式相似的问题：\n"
        for i, pattern in enumerate(recent_patterns[:10], 1):
            avoid_text += f"{i}. 模式ID: {pattern}\n"
        
        return avoid_text

    def _get_role_style_guide(self, batch_size: int) -> str:
        """获取角色和风格分布指导"""
        guide = "本批次分布目标：\n"
        
        for role, ratio in self.user_roles.items():
            current = self.state['role_count'][role]
            target = int(batch_size * ratio)
            guide += f"- {role}: {target}条 (当前总数: {current})\n"
        
        guide += "\n风格分布：\n"
        for style, ratio in self.language_styles.items():
            current = self.state['style_count'][style]
            target = int(batch_size * ratio)
            guide += f"- {style}: {target}条 (当前总数: {current})\n"
            
        return guide
    
    def _calculate_tool_targets(self) -> Dict[str, int]:
        """根据总对话数动态计算各工具的目标数量"""
        targets = {}
        for tool, ratio in self.tool_ratios.items():
            targets[tool] = int(self.total_conversations * ratio)
        
        # 确保总数正确（处理舍入误差）
        current_total = sum(targets.values())
        if current_total != self.total_conversations:
            # 将差值分配给第一个工具
            first_tool = list(targets.keys())[0]
            targets[first_tool] += self.total_conversations - current_total
        
        return targets

    def _format_priority_tools(self, priority_tools: List) -> str:
        """格式化优先工具信息"""
        if not priority_tools:
            return "所有工具已完成目标"
        
        result = ""
        for tool, allocation in priority_tools:
            current = self.state['generated_count'][tool]
            target = self.tool_targets[tool]
            result += f"- {tool}: 分配{allocation}条 (当前{current}/{target})\n"
        
        return result

    def record_generated(self, conversations: List[Dict], tool_name: str, 
                        user_role: str = None, language_style: str = None):
        """记录已生成的数据"""
        
        # 记录问题特征
        for conv in conversations:
            if conv.get("from") == "user":
                question = conv.get("value", "")
                signature = self.get_question_signature(question)
                self.state["used_questions"].add(signature)
                
                # 提取并记录参数
                self._extract_parameters(question)
        
        # 更新工具计数
        self.state["generated_count"][tool_name] += 1
        
        # 更新角色和风格计数
        if user_role:
            self.state["role_count"][user_role] += 1
        if language_style:
            self.state["style_count"][language_style] += 1
            
        # 更新总数
        self.state["total_generated"] += 1
        
        # 保存状态
        self._save_state()

    def _extract_parameters(self, question: str):
        """从问题中提取参数"""
        # 提取地址
        addresses = re.findall(r'0x[a-fA-F0-9]{40}', question)
        for addr in addresses:
            self.state["used_parameters"]["addresses"].add(addr)
        
        # 提取交易哈希
        tx_hashes = re.findall(r'0x[a-fA-F0-9]{64}', question)
        for tx in tx_hashes:
            self.state["used_parameters"]["tx_hashes"].add(tx)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "总进度": f"{self.state['total_generated']}/{self.total_conversations}",
            "完成率": f"{self.state['total_generated']/self.total_conversations*100:.1f}%",
            "工具进度": {},
            "角色分布": dict(self.state["role_count"]),
            "风格分布": dict(self.state["style_count"]),
            "已用问题模式": len(self.state["used_questions"]),
            "已用地址": len(self.state["used_parameters"]["addresses"]),
            "已用交易哈希": len(self.state["used_parameters"]["tx_hashes"])
        }
        
        for tool, target in self.tool_targets.items():
            current = self.state["generated_count"][tool]
            stats["工具进度"][tool] = f"{current}/{target} ({current/target*100:.1f}%)"
        
        return stats


# 使用示例
if __name__ == "__main__":
    # 测试不同的total_conversations配置
    print("=== 测试 100 个对话的配置 ===")
    manager = DatasetDedupManager(total_conversations=100)
    
    # 获取下一批次的动态prompt
    batch_prompt = manager.get_next_batch_prompt(10)
    print("=== 下一批次动态Prompt ===")
    print(batch_prompt)
    
    # 查看统计信息
    stats = manager.get_statistics()
    print("\n=== 当前统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== 工具目标分配 ===")
    for tool, target in list(manager.tool_targets.items())[:5]:
        print(f"{tool}: {target}")