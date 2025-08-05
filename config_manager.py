#!/usr/bin/env python3
"""
配置管理器
负责读取和管理配置文件
"""
import json
import os
from typing import Dict, Any


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_file):
                print(f"⚠️ 配置文件 {self.config_file} 不存在，使用默认配置")
                return self._get_default_config()
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"✅ 已加载配置文件: {self.config_file}")
                return config
                
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            print("使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "api": {
                "deepseek_api_key": "",
                "base_url": "https://api.deepseek.com",
                "timeout": 60
            },
            "generation": {
                "default_total_conversations": 6000,
                "default_batch_size": 50,
                "default_prompt_file": "prompt.txt",
                "default_output_file": "function_calling_dataset_smart.json"
            },
            "system": {
                "enable_debug": False,
                "auto_cleanup_temp_files": True,
                "max_retries": 3
            }
        }
    
    def get_api_key(self) -> str:
        """获取API Key"""
        api_key = self.config.get("api", {}).get("deepseek_api_key", "")
        
        if not api_key:
            print("⚠️ 配置文件中未找到API Key")
            # 尝试从环境变量获取
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if api_key:
                print("✅ 从环境变量 DEEPSEEK_API_KEY 获取API Key")
            else:
                print("❌ 未找到API Key，请在config.json中设置或设置环境变量DEEPSEEK_API_KEY")
        
        return api_key
    
    def get_api_config(self) -> Dict[str, Any]:
        """获取API配置"""
        return self.config.get("api", {})
    
    def get_generation_config(self) -> Dict[str, Any]:
        """获取生成配置"""
        return self.config.get("generation", {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.config.get("system", {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项（支持点号分隔的嵌套键）"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
                print(f"✅ 配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
    
    def set_api_key(self, api_key: str):
        """设置API Key"""
        if "api" not in self.config:
            self.config["api"] = {}
        self.config["api"]["deepseek_api_key"] = api_key
        print("✅ API Key已更新")
    
    def validate_config(self) -> bool:
        """验证配置的完整性"""
        required_keys = [
            "api.deepseek_api_key",
            "generation.default_total_conversations",
            "generation.default_batch_size"
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"⚠️ 配置不完整，缺少: {', '.join(missing_keys)}")
            return False
        
        # 验证API Key
        api_key = self.get_api_key()
        if not api_key:
            print("⚠️ API Key未配置")
            return False
        
        print("✅ 配置验证通过")
        return True


# 使用示例
if __name__ == "__main__":
    config = ConfigManager()
    
    print("=== 配置信息 ===")
    print(f"API Key: {config.get_api_key()[:8]}..." if config.get_api_key() else "未设置")
    print(f"默认对话数量: {config.get('generation.default_total_conversations')}")
    print(f"默认批次大小: {config.get('generation.default_batch_size')}")
    print(f"调试模式: {config.get('system.enable_debug')}")
    
    # 验证配置
    config.validate_config()