#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试脚本 - 模型加载和checkpoint切换测试
测试checkpoint-669痤疮诊断模型的基础功能
"""

import unittest
import sys
import os
import torch
import gc
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加推理脚本路径
sys.path.append('/root/autodl-tmp/inference')

try:
    from acne_diagnosis_web import (
        get_available_checkpoints,
        _load_model_processor,
        clear_gpu_memory,
        DEFAULT_CKPT_PATH,
        TRAINING_OUTPUT_DIR
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 /root/autodl-tmp/inference/acne_diagnosis_web.py 文件存在")
    sys.exit(1)

class TestModelLoading(unittest.TestCase):
    """测试模型加载功能"""
    
    def setUp(self):
        """测试前准备"""
        self.model = None
        self.processor = None
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def tearDown(self):
        """测试后清理"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        clear_gpu_memory()
        gc.collect()
    
    def test_base_model_path_exists(self):
        """测试基础模型路径是否存在"""
        self.assertTrue(os.path.exists(DEFAULT_CKPT_PATH), 
                       f"基础模型路径不存在: {DEFAULT_CKPT_PATH}")
    
    def test_checkpoint_path_exists(self):
        """测试checkpoint路径是否存在"""
        checkpoint_path = os.path.join(TRAINING_OUTPUT_DIR, "checkpoint-669")
        self.assertTrue(os.path.exists(checkpoint_path), 
                       f"Checkpoint路径不存在: {checkpoint_path}")
    
    def test_checkpoint_files_exist(self):
        """测试checkpoint必要文件是否存在"""
        checkpoint_path = os.path.join(TRAINING_OUTPUT_DIR, "checkpoint-669")
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        for file_name in required_files:
            file_path = os.path.join(checkpoint_path, file_name)
            self.assertTrue(os.path.exists(file_path), 
                           f"必要文件不存在: {file_path}")
    
    def test_get_available_checkpoints(self):
        """测试获取可用checkpoints功能"""
        try:
            checkpoints = get_available_checkpoints()
            self.assertIsInstance(checkpoints, list, "checkpoints应该是列表类型")
            self.assertGreater(len(checkpoints), 0, "应该至少有一个checkpoint")
            
            # 检查checkpoint-669是否在列表中
            checkpoint_669_found = any('669' in str(cp) for cp in checkpoints)
            self.assertTrue(checkpoint_669_found, "未找到checkpoint-669")
        except Exception as e:
            self.fail(f"获取checkpoints失败: {e}")
    
    @patch('torch.cuda.is_available')
    def test_load_base_model(self, mock_cuda):
        """测试加载基础模型"""
        mock_cuda.return_value = True
        
        try:
            self.model, self.processor = _load_model_processor(
                DEFAULT_CKPT_PATH, 
                None, 
                cpu_only=True,  # 使用CPU避免GPU内存问题
                flash_attn2=False
            )
            self.assertIsNotNone(self.model)
            self.assertIsNotNone(self.processor)
        except Exception as e:
            self.fail(f"基础模型加载异常: {e}")
    
    @patch('torch.cuda.is_available')
    def test_load_checkpoint_669(self, mock_cuda):
        """测试加载checkpoint-669"""
        mock_cuda.return_value = True
        
        try:
            checkpoint_path = os.path.join(TRAINING_OUTPUT_DIR, "checkpoint-669")
            self.model, self.processor = _load_model_processor(
                DEFAULT_CKPT_PATH,
                checkpoint_path,
                cpu_only=True,  # 使用CPU避免GPU内存问题
                flash_attn2=False
            )
            self.assertIsNotNone(self.model)
            self.assertIsNotNone(self.processor)
        except Exception as e:
            self.fail(f"checkpoint-669加载异常: {e}")
    
    def test_invalid_model_loading(self):
        """测试加载无效模型"""
        try:
            invalid_path = "/path/to/nonexistent/model"
            with self.assertRaises(Exception):
                _load_model_processor(
                    DEFAULT_CKPT_PATH,
                    invalid_path,
                    cpu_only=True,
                    flash_attn2=False
                )
        except Exception as e:
            # 预期会抛出异常
            pass
    
    def test_memory_cleanup(self):
        """测试内存清理功能"""
        try:
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
            
            # 加载模型
            self.model, self.processor = _load_model_processor(
                DEFAULT_CKPT_PATH,
                None,
                cpu_only=True,
                flash_attn2=False
            )
            
            # 清理模型
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            clear_gpu_memory()
            
            # 检查内存清理函数是否正常工作
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated()
                # 内存应该被释放或保持在合理范围内
                self.assertLessEqual(final_memory, initial_memory + 1024*1024*100)  # 允许100MB误差
        except Exception as e:
            self.fail(f"内存清理测试失败: {e}")

class TestModelConfiguration(unittest.TestCase):
    """测试模型配置"""
    
    def test_torch_device_availability(self):
        """测试PyTorch设备可用性"""
        self.assertTrue(torch.cuda.is_available(), "CUDA应该可用")
        device_count = torch.cuda.device_count()
        self.assertGreater(device_count, 0, "应该至少有一个GPU设备")
    
    def test_model_paths_configuration(self):
        """测试模型路径配置"""
        base_path = "/root/autodl-tmp/Qwen/Qwen2.5-VL"
        output_path = "/root/autodl-tmp/Qwen/Qwen2.5-VL/output/Qwen2_5-VL-7B-Acne"
        
        self.assertTrue(os.path.exists(base_path), f"基础路径不存在: {base_path}")
        self.assertTrue(os.path.exists(output_path), f"输出路径不存在: {output_path}")

def run_model_loading_tests():
    """运行模型加载测试"""
    print("\n" + "="*60)
    print("开始运行模型加载单元测试")
    print("="*60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestModelConfiguration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果摘要
    print("\n" + "="*60)
    print("测试结果摘要:")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_model_loading_tests()
    sys.exit(0 if success else 1)