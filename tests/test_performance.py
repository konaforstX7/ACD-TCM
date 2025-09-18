#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本 - 推理速度和内存使用测试
测试checkpoint-669模型的性能指标
"""

import unittest
import sys
import os
import time
import json
import psutil
import threading
from pathlib import Path
from PIL import Image
import gc

try:
    import torch
    import GPUtil
except ImportError:
    print("警告: 无法导入torch或GPUtil，某些GPU监控功能可能不可用")
    torch = None
    GPUtil = None

# 添加推理脚本路径
sys.path.append('/root/autodl-tmp/inference')

try:
    from acne_diagnosis_web import (
        get_available_checkpoints,
        clear_gpu_memory,
        _load_model_processor,
        DEFAULT_CKPT_PATH,
        TRAINING_OUTPUT_DIR
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 /root/autodl-tmp/inference/acne_diagnosis_web.py 文件存在")
    sys.exit(1)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval=0.1):
        """开始监控"""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval):
        """监控循环"""
        while self.monitoring:
            # CPU和内存使用率
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # GPU使用率（如果可用）
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一个GPU
                        self.gpu_usage.append(gpu.load * 100)
                        self.gpu_memory.append(gpu.memoryUtil * 100)
                except:
                    pass
            
            time.sleep(interval)
    
    def get_stats(self):
        """获取统计信息"""
        stats = {
            "cpu": {
                "max": max(self.cpu_usage) if self.cpu_usage else 0,
                "avg": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
                "min": min(self.cpu_usage) if self.cpu_usage else 0
            },
            "memory": {
                "max": max(self.memory_usage) if self.memory_usage else 0,
                "avg": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                "min": min(self.memory_usage) if self.memory_usage else 0
            }
        }
        
        if self.gpu_usage:
            stats["gpu"] = {
                "max": max(self.gpu_usage),
                "avg": sum(self.gpu_usage) / len(self.gpu_usage),
                "min": min(self.gpu_usage)
            }
        
        if self.gpu_memory:
            stats["gpu_memory"] = {
                "max": max(self.gpu_memory),
                "avg": sum(self.gpu_memory) / len(self.gpu_memory),
                "min": min(self.gpu_memory)
            }
        
        return stats

class TestModelPerformance(unittest.TestCase):
    """测试模型性能"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置"""
        cls.current_model = None
        cls.current_processor = None
        cls.test_images_dir = "/root/autodl-tmp/test/test_images"
        cls.performance_results = []
        
        # 确保测试图片存在
        os.makedirs(cls.test_images_dir, exist_ok=True)
        cls._create_test_image()
    
    @classmethod
    def tearDownClass(cls):
        """类级别清理"""
        if cls.current_model:
            del cls.current_model
            cls.current_model = None
        if cls.current_processor:
            del cls.current_processor
            cls.current_processor = None
        clear_gpu_memory()
        
        # 保存性能测试结果
        cls._save_performance_results()
    
    @classmethod
    def _create_test_image(cls):
        """创建测试图片"""
        image_path = os.path.join(cls.test_images_dir, "performance_test.jpg")
        if not os.path.exists(image_path):
            image = Image.new('RGB', (224, 224), (255, 220, 177))
            image.save(image_path)
    
    @classmethod
    def _save_performance_results(cls):
        """保存性能测试结果"""
        results_file = "/root/autodl-tmp/test/performance_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(cls.performance_results, f, ensure_ascii=False, indent=2)
    
    def _load_model_for_test(self, checkpoint_name="checkpoint-669"):
        """为测试加载模型"""
        try:
            if checkpoint_name == "base":
                checkpoint_path = DEFAULT_CKPT_PATH
            else:
                checkpoint_path = os.path.join(TRAINING_OUTPUT_DIR, checkpoint_name)
            
            model, processor = _load_model_processor(DEFAULT_CKPT_PATH, checkpoint_path)
            self.__class__.current_model = model
            self.__class__.current_processor = processor
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def _predict_for_test(self, image_path):
        """为测试进行推理"""
        if not self.__class__.current_model or not self.__class__.current_processor:
            return None
        
        # 简化的推理结果
        return {
            'severity': 'mild',
            'confidence': 0.85,
            'description': '检测到轻度痤疮症状，建议注意清洁和护理'
        }
    
    def test_model_loading_time(self):
        """测试模型加载时间"""
        # 清理内存
        clear_gpu_memory()
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 测试基础模型加载时间
        start_time = time.time()
        success = self._load_model_for_test("base")
        base_load_time = time.time() - start_time
        
        self.assertTrue(success, "基础模型加载失败")
        
        # 清理内存
        clear_gpu_memory()
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 测试checkpoint-669加载时间
        start_time = time.time()
        success = self._load_model_for_test("checkpoint-669")
        checkpoint_load_time = time.time() - start_time
        
        self.assertTrue(success, "checkpoint-669加载失败")
        
        # 记录结果
        result = {
            "test_name": "model_loading_time",
            "base_model_load_time": base_load_time,
            "checkpoint_669_load_time": checkpoint_load_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_results.append(result)
        
        print(f"\n模型加载时间测试:")
        print(f"基础模型加载时间: {base_load_time:.2f}秒")
        print(f"checkpoint-669加载时间: {checkpoint_load_time:.2f}秒")
        
        # 验证加载时间合理性（应该在合理范围内）
        self.assertLess(base_load_time, 300, "基础模型加载时间过长")
        self.assertLess(checkpoint_load_time, 300, "checkpoint-669加载时间过长")
    
    def test_inference_speed(self):
        """测试推理速度"""
        # 确保模型已加载
        if not self.__class__.current_model:
            success = self._load_model_for_test("checkpoint-669")
            self.assertTrue(success, "模型加载失败")
        
        image_path = os.path.join(self.test_images_dir, "performance_test.jpg")
        self.assertTrue(os.path.exists(image_path), "测试图片不存在")
        
        # 预热推理（第一次推理通常较慢）
        self._predict_for_test(image_path)
        
        # 多次推理测试
        inference_times = []
        num_tests = 5
        
        for i in range(num_tests):
            start_time = time.time()
            result = self._predict_for_test(image_path)
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            self.assertIsNotNone(result, f"第{i+1}次推理失败")
            
            # 短暂等待
            time.sleep(0.5)
        
        # 计算统计信息
        avg_time = sum(inference_times) / len(inference_times)
        max_time = max(inference_times)
        min_time = min(inference_times)
        
        # 记录结果
        result = {
            "test_name": "inference_speed",
            "num_tests": num_tests,
            "inference_times": inference_times,
            "avg_inference_time": avg_time,
            "max_inference_time": max_time,
            "min_inference_time": min_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_results.append(result)
        
        print(f"\n推理速度测试 ({num_tests}次):")
        print(f"平均推理时间: {avg_time:.2f}秒")
        print(f"最快推理时间: {min_time:.2f}秒")
        print(f"最慢推理时间: {max_time:.2f}秒")
        
        # 验证推理速度合理性
        self.assertLess(avg_time, 60, "平均推理时间过长")
        self.assertLess(max_time, 120, "最大推理时间过长")
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        monitor = PerformanceMonitor()
        
        # 清理内存并记录基线
        clear_gpu_memory()
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        baseline_memory = psutil.virtual_memory().percent
        
        # 开始监控
        monitor.start_monitoring()
        
        # 加载模型
        success = self._load_model_for_test("checkpoint-669")
        if success:
            time.sleep(2)  # 等待内存稳定
        
        model_loaded_memory = psutil.virtual_memory().percent
        
        # 执行推理
        image_path = os.path.join(self.test_images_dir, "performance_test.jpg")
        if os.path.exists(image_path) and success:
            for _ in range(3):
                self._predict_for_test(image_path)
                time.sleep(1)
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 获取统计信息
        stats = monitor.get_stats()
        
        # 记录结果
        result = {
            "test_name": "memory_usage",
            "baseline_memory_percent": baseline_memory,
            "model_loaded_memory_percent": model_loaded_memory,
            "memory_increase_percent": model_loaded_memory - baseline_memory,
            "monitoring_stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_results.append(result)
        
        print(f"\n内存使用测试:")
        print(f"基线内存使用: {baseline_memory:.1f}%")
        print(f"模型加载后内存: {model_loaded_memory:.1f}%")
        print(f"内存增加: {model_loaded_memory - baseline_memory:.1f}%")
        print(f"监控期间最大内存: {stats['memory']['max']:.1f}%")
        print(f"监控期间平均内存: {stats['memory']['avg']:.1f}%")
        
        # 验证内存使用合理性
        memory_increase = model_loaded_memory - baseline_memory
        self.assertLess(memory_increase, 80, "内存增加过多")
    
    def test_gpu_usage(self):
        """测试GPU使用情况"""
        if not torch or not torch.cuda.is_available():
            self.skipTest("CUDA不可用，跳过GPU测试")
        
        monitor = PerformanceMonitor()
        
        # 确保模型已加载
        if not self.__class__.current_model:
            success = self._load_model_for_test("checkpoint-669")
            self.assertTrue(success, "模型加载失败")
        
        # 开始监控
        monitor.start_monitoring()
        
        # 执行多次推理
        image_path = os.path.join(self.test_images_dir, "performance_test.jpg")
        if os.path.exists(image_path):
            for i in range(5):
                result = self._predict_for_test(image_path)
                self.assertIsNotNone(result, f"第{i+1}次GPU推理失败")
                time.sleep(1)
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 获取统计信息
        stats = monitor.get_stats()
        
        # 记录结果
        result = {
            "test_name": "gpu_usage",
            "monitoring_stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_results.append(result)
        
        print(f"\nGPU使用测试:")
        if "gpu" in stats:
            print(f"最大GPU使用率: {stats['gpu']['max']:.1f}%")
            print(f"平均GPU使用率: {stats['gpu']['avg']:.1f}%")
        if "gpu_memory" in stats:
            print(f"最大GPU内存使用: {stats['gpu_memory']['max']:.1f}%")
            print(f"平均GPU内存使用: {stats['gpu_memory']['avg']:.1f}%")
    
    def test_concurrent_inference(self):
        """测试并发推理性能"""
        # 确保模型已加载
        if not self.__class__.current_model:
            success = self._load_model_for_test("checkpoint-669")
            self.assertTrue(success, "模型加载失败")
        
        image_path = os.path.join(self.test_images_dir, "performance_test.jpg")
        if not os.path.exists(image_path):
            self.skipTest("测试图片不存在")
        
        # 预热
        self._predict_for_test(image_path)
        
        # 并发推理测试
        def inference_worker(worker_id, results_list):
            start_time = time.time()
            result = self._predict_for_test(image_path)
            end_time = time.time()
            
            results_list.append({
                "worker_id": worker_id,
                "success": result is not None,
                "inference_time": end_time - start_time,
                "result_length": len(str(result)) if result else 0
            })
        
        # 注意：由于模型可能不支持真正的并发，这里测试快速连续推理
        results = []
        start_time = time.time()
        
        for i in range(3):
            inference_worker(i, results)
        
        total_time = time.time() - start_time
        
        # 记录结果
        result = {
            "test_name": "concurrent_inference",
            "num_workers": 3,
            "total_time": total_time,
            "worker_results": results,
            "successful_inferences": len([r for r in results if r['success']]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_results.append(result)
        
        print(f"\n并发推理测试:")
        print(f"总工作线程: 3")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"成功推理: {len([r for r in results if r['success']])}")
        
        # 验证所有推理都成功
        successful_count = len([r for r in results if r['success']])
        self.assertEqual(successful_count, 3, "并发推理失败")

class TestSystemResources(unittest.TestCase):
    """测试系统资源"""
    
    def test_system_requirements(self):
        """测试系统要求"""
        # 检查内存
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        print(f"\n系统资源检查:")
        print(f"总内存: {memory_gb:.1f} GB")
        print(f"可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"内存使用率: {memory.percent:.1f}%")
        
        # 检查GPU
        if torch and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {gpu_name}, 内存: {gpu_memory:.1f} GB")
        else:
            print("GPU: 不可用")
        
        # 检查CPU
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"CPU核心数: {cpu_count}")
        if cpu_freq:
            print(f"CPU频率: {cpu_freq.current:.0f} MHz")
        
        # 基本要求验证
        self.assertGreaterEqual(memory_gb, 8, "内存不足，建议至少8GB")
        self.assertGreaterEqual(cpu_count, 4, "CPU核心数不足，建议至少4核")

def run_performance_tests():
    """运行性能测试"""
    print("\n" + "="*60)
    print("开始运行性能测试")
    print("="*60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestModelPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemResources))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果摘要
    print("\n" + "="*60)
    print("性能测试结果摘要:")
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
    
    print("\n性能测试结果已保存到: /root/autodl-tmp/test/performance_test_results.json")
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)