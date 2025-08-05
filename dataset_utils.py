#!/usr/bin/env python3
"""
数据集处理工具类
提供JSON解析、文件读写等工具函数
"""
import json
import re
from typing import List, Dict

class DatasetUtils:
    """数据集处理工具类"""
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return ""
    
    @staticmethod
    def write_json_file(data: List[Dict], file_path: str):
        """写入JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"数据已保存到: {file_path}")
        except Exception as e:
            print(f"写入文件失败 {file_path}: {e}")
    
    @staticmethod
    def load_json_file(file_path: str) -> List[Dict]:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"文件不存在: {file_path}")
            return []
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return []
    
    @staticmethod
    def extract_json_from_response(response_text: str) -> List[Dict]:
        """从响应文本中提取JSON数据，返回包含conversations字段的对话对象列表"""
        try:
            # 首先尝试直接解析整个响应
            try:
                data = json.loads(response_text)
                if isinstance(data, list):
                    # 如果是对话对象列表，直接返回
                    if data and isinstance(data[0], dict) and "conversations" in data[0]:
                        return data
                    # 如果是消息列表，包装成对话对象
                    else:
                        return [{"conversations": data}]
                elif isinstance(data, dict) and "conversations" in data:
                    # 单个对话对象，包装成列表
                    return [data]
            except json.JSONDecodeError:
                pass
            
            # 寻找JSON代码块 ```json ... ```
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            all_conversation_objects = []
            
            for block in json_blocks:
                try:
                    data = json.loads(block)
                    if isinstance(data, list):
                        # 检查是否已经是对话对象列表
                        if data and isinstance(data[0], dict) and "conversations" in data[0]:
                            all_conversation_objects.extend(data)
                        else:
                            # 消息列表，包装成对话对象
                            all_conversation_objects.append({"conversations": data})
                    elif isinstance(data, dict) and "conversations" in data:
                        # 单个对话对象
                        all_conversation_objects.append(data)
                except json.JSONDecodeError:
                    continue
            
            if all_conversation_objects:
                return all_conversation_objects
            
            # 最后尝试寻找普通的JSON数组
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > 0:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                # 包装成对话对象
                return [{"conversations": data}]
            
            print("未找到有效的JSON数据")
            return []
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return []
        except Exception as e:
            print(f"提取JSON失败: {e}")
            return []
    
    @staticmethod
    def convert_to_sharegpt_format(conversations: List[Dict]) -> List[Dict]:
        """将对话数据转换为ShareGPT格式"""
        sharegpt_data = []
        
        for conv in conversations:
            if "conversations" in conv:
                # 已经是conversations格式
                sharegpt_data.append(conv)
            else:
                # 需要转换格式
                sharegpt_data.append({"conversations": conv})
        
        return sharegpt_data