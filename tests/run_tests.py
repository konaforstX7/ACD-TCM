#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint-669 痤疮诊断模型自动化测试运行脚本

功能:
- 运行所有测试套件（单元测试、集成测试、性能测试）
- 生成详细的测试报告
- 支持不同的测试场景（冒烟测试、完整回归测试等）
- 收集系统信息和性能指标
- 生成HTML和Markdown格式的报告
"""

import os
import sys
import json
import time
import unittest
import argparse
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# 添加项目路径
sys.path.append('/root/autodl-tmp')
sys.path.append('/root/autodl-tmp/inference')

try:
    import torch
    import psutil
    import GPUtil
except ImportError as e:
    print(f"警告: 无法导入某些依赖包: {e}")
    print("请确保已安装所有必要的依赖包")

class TestRunner:
    """测试运行器类"""
    
    def __init__(self, test_dir: str = "/root/autodl-tmp/test"):
        self.test_dir = Path(test_dir)
        self.results_dir = self.test_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 加载测试配置
        self.config = self.load_test_config()
        
        # 测试结果存储
        self.test_results = {
            "summary": {},
            "unit_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "system_info": {},
            "errors": [],
            "start_time": None,
            "end_time": None,
            "duration": None
        }
        
    def load_test_config(self) -> Dict[str, Any]:
        """加载测试配置文件"""
        config_file = self.test_dir / "test_cases.json"
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 无法加载测试配置文件: {e}")
            return {}
    
    def collect_system_info(self) -> Dict[str, Any]:
        """收集系统信息"""
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation()
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            }
        }
        
        # 收集GPU信息
        try:
            if torch.cuda.is_available():
                system_info["gpu"] = {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                }
                
                # 获取GPU详细信息
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = []
                    for gpu in gpus:
                        gpu_info.append({
                            "id": gpu.id,
                            "name": gpu.name,
                            "memory_total_mb": gpu.memoryTotal,
                            "memory_used_mb": gpu.memoryUsed,
                            "memory_free_mb": gpu.memoryFree,
                            "temperature": gpu.temperature,
                            "load": gpu.load
                        })
                    system_info["gpu"]["details"] = gpu_info
            else:
                system_info["gpu"] = {"cuda_available": False}
        except Exception as e:
            system_info["gpu"] = {"error": str(e)}
        
        # 收集Python包版本信息
        try:
            import torch
            import transformers
            import peft
            from PIL import Image
            
            system_info["packages"] = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "pillow": Image.__version__ if hasattr(Image, '__version__') else "Unknown",
                "psutil": psutil.__version__
            }
        except Exception as e:
            system_info["packages"] = {"error": str(e)}
        
        return system_info
    
    def run_test_suite(self, test_file: str, test_class: Optional[str] = None) -> Dict[str, Any]:
        """运行指定的测试套件"""
        test_path = self.test_dir / test_file
        if not test_path.exists():
            return {
                "status": "error",
                "message": f"测试文件不存在: {test_file}",
                "tests": [],
                "duration": 0
            }
        
        print(f"\n运行测试套件: {test_file}")
        start_time = time.time()
        
        try:
            # 动态导入测试模块
            spec = importlib.util.spec_from_file_location("test_module", test_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # 创建测试套件
            loader = unittest.TestLoader()
            if test_class:
                suite = loader.loadTestsFromTestCase(getattr(test_module, test_class))
            else:
                suite = loader.loadTestsFromModule(test_module)
            
            # 运行测试
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            result = runner.run(suite)
            
            duration = time.time() - start_time
            
            # 收集测试结果
            test_results = {
                "status": "passed" if result.wasSuccessful() else "failed",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
                "duration": round(duration, 2),
                "details": {
                    "failures": [{
                        "test": str(test),
                        "traceback": traceback
                    } for test, traceback in result.failures],
                    "errors": [{
                        "test": str(test),
                        "traceback": traceback
                    } for test, traceback in result.errors]
                }
            }
            
            return test_results
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"运行测试套件时发生错误: {str(e)}"
            print(f"错误: {error_msg}")
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": error_msg,
                "duration": round(duration, 2),
                "traceback": traceback.format_exc()
            }
    
    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """运行指定的测试场景"""
        if "test_scenarios" not in self.config:
            return {"status": "error", "message": "测试配置中未找到测试场景"}
        
        scenarios = self.config["test_scenarios"]
        if scenario_name not in scenarios:
            return {"status": "error", "message": f"未找到测试场景: {scenario_name}"}
        
        scenario = scenarios[scenario_name]
        print(f"\n运行测试场景: {scenario_name}")
        print(f"描述: {scenario.get('description', '')}")
        
        scenario_start = time.time()
        scenario_results = {
            "name": scenario_name,
            "description": scenario.get("description", ""),
            "status": "passed",
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            }
        }
        
        tests_to_run = scenario.get("tests", [])
        if tests_to_run == "all":
            # 运行所有测试
            tests_to_run = [
                "unit_tests",
                "integration_tests", 
                "performance_tests"
            ]
        
        for test_spec in tests_to_run:
            if isinstance(test_spec, str):
                if test_spec == "unit_tests":
                    result = self.run_test_suite("test_model_loading.py")
                elif test_spec == "integration_tests":
                    result = self.run_test_suite("test_acne_diagnosis.py")
                elif test_spec == "performance_tests":
                    result = self.run_test_suite("test_performance.py")
                else:
                    # 解析具体的测试用例
                    parts = test_spec.split(".")
                    if len(parts) == 2:
                        suite_name, test_name = parts
                        if suite_name == "unit_tests":
                            result = self.run_test_suite("test_model_loading.py")
                        elif suite_name == "integration_tests":
                            result = self.run_test_suite("test_acne_diagnosis.py")
                        elif suite_name == "performance_tests":
                            result = self.run_test_suite("test_performance.py")
                        else:
                            continue
                    else:
                        continue
                
                scenario_results["tests"].append({
                    "name": test_spec,
                    "result": result
                })
                
                # 更新摘要
                if result.get("status") == "passed":
                    scenario_results["summary"]["passed"] += result.get("tests_run", 0)
                else:
                    scenario_results["summary"]["failed"] += result.get("failures", 0)
                    scenario_results["summary"]["errors"] += result.get("errors", 0)
                    scenario_results["status"] = "failed"
                
                scenario_results["summary"]["total"] += result.get("tests_run", 0)
        
        scenario_results["duration"] = round(time.time() - scenario_start, 2)
        return scenario_results
    
    def generate_report(self, output_format: str = "both") -> None:
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成JSON报告
        json_file = self.results_dir / f"test_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n测试报告已生成: {json_file}")
        
        # 生成Markdown报告
        if output_format in ["markdown", "both"]:
            self.generate_markdown_report(timestamp)
        
        # 生成HTML报告
        if output_format in ["html", "both"]:
            self.generate_html_report(timestamp)
    
    def generate_markdown_report(self, timestamp: str) -> None:
        """生成Markdown格式的报告"""
        template_file = self.test_dir / "test_report_template.md"
        if not template_file.exists():
            print("警告: 未找到报告模板文件")
            return
        
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # 替换模板变量
        replacements = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_version": "1.0",
            "test_environment": f"{platform.system()} {platform.release()}",
            "tester_name": "自动化测试",
            "total_tests": str(self.test_results["summary"].get("total", 0)),
            "passed_tests": str(self.test_results["summary"].get("passed", 0)),
            "failed_tests": str(self.test_results["summary"].get("failed", 0)),
            "skipped_tests": str(self.test_results["summary"].get("skipped", 0)),
            "success_rate": str(round(self.test_results["summary"].get("success_rate", 0), 1)),
            "total_duration": f"{self.test_results.get('duration', 0):.2f}s"
        }
        
        # 添加系统信息
        system_info = self.test_results.get("system_info", {})
        replacements.update({
            "cpu_info": system_info.get("platform", {}).get("processor", "Unknown"),
            "memory_info": f"{system_info.get('hardware', {}).get('memory_total_gb', 0)}GB",
            "gpu_info": system_info.get("gpu", {}).get("device_name", "Unknown"),
            "os_info": f"{system_info.get('platform', {}).get('system', '')} {system_info.get('platform', {}).get('release', '')}",
            "python_version": system_info.get("python", {}).get("version", "Unknown")
        })
        
        # 替换所有变量
        for key, value in replacements.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        # 保存报告
        md_file = self.results_dir / f"test_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(template)
        
        print(f"Markdown报告已生成: {md_file}")
    
    def generate_html_report(self, timestamp: str) -> None:
        """生成HTML格式的报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checkpoint-669 测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .test-section {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .error {{ color: orange; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .json-data {{ background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Checkpoint-669 痤疮诊断模型测试报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>测试摘要</h2>
        <p>总测试数: {self.test_results['summary'].get('total', 0)}</p>
        <p class="passed">通过: {self.test_results['summary'].get('passed', 0)}</p>
        <p class="failed">失败: {self.test_results['summary'].get('failed', 0)}</p>
        <p>成功率: {self.test_results['summary'].get('success_rate', 0):.1f}%</p>
        <p>总耗时: {self.test_results.get('duration', 0):.2f}秒</p>
    </div>
    
    <div class="test-section">
        <h2>详细结果</h2>
        <div class="json-data">
            <pre>{json.dumps(self.test_results, ensure_ascii=False, indent=2)}</pre>
        </div>
    </div>
</body>
</html>
        """
        
        html_file = self.results_dir / f"test_report_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {html_file}")
    
    def run_all_tests(self, scenario: str = "full_regression") -> None:
        """运行所有测试"""
        print(f"开始运行测试场景: {scenario}")
        print("=" * 60)
        
        self.test_results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        # 收集系统信息
        print("收集系统信息...")
        self.test_results["system_info"] = self.collect_system_info()
        
        try:
            # 运行指定场景
            if scenario in self.config.get("test_scenarios", {}):
                scenario_result = self.run_scenario(scenario)
                self.test_results["scenario"] = scenario_result
                
                # 更新总体摘要
                self.test_results["summary"] = scenario_result["summary"]
                self.test_results["summary"]["success_rate"] = (
                    (scenario_result["summary"]["passed"] / max(scenario_result["summary"]["total"], 1)) * 100
                )
            else:
                # 运行所有测试套件
                print("\n运行单元测试...")
                self.test_results["unit_tests"] = self.run_test_suite("test_model_loading.py")
                
                print("\n运行集成测试...")
                self.test_results["integration_tests"] = self.run_test_suite("test_acne_diagnosis.py")
                
                print("\n运行性能测试...")
                self.test_results["performance_tests"] = self.run_test_suite("test_performance.py")
                
                # 计算总体摘要
                total_tests = 0
                total_passed = 0
                total_failed = 0
                total_errors = 0
                
                for suite_name in ["unit_tests", "integration_tests", "performance_tests"]:
                    suite_result = self.test_results.get(suite_name, {})
                    if suite_result.get("status") != "error":
                        total_tests += suite_result.get("tests_run", 0)
                        total_failed += suite_result.get("failures", 0)
                        total_errors += suite_result.get("errors", 0)
                        total_passed += (suite_result.get("tests_run", 0) - 
                                       suite_result.get("failures", 0) - 
                                       suite_result.get("errors", 0))
                
                self.test_results["summary"] = {
                    "total": total_tests,
                    "passed": total_passed,
                    "failed": total_failed,
                    "errors": total_errors,
                    "success_rate": (total_passed / max(total_tests, 1)) * 100
                }
        
        except Exception as e:
            error_msg = f"运行测试时发生错误: {str(e)}"
            print(f"错误: {error_msg}")
            self.test_results["errors"].append({
                "message": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
        
        # 记录结束时间
        end_time = time.time()
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["duration"] = round(end_time - start_time, 2)
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("测试完成摘要:")
        print(f"总测试数: {self.test_results['summary'].get('total', 0)}")
        print(f"通过: {self.test_results['summary'].get('passed', 0)}")
        print(f"失败: {self.test_results['summary'].get('failed', 0)}")
        print(f"错误: {self.test_results['summary'].get('errors', 0)}")
        print(f"成功率: {self.test_results['summary'].get('success_rate', 0):.1f}%")
        print(f"总耗时: {self.test_results['duration']:.2f}秒")
        print("=" * 60)
        
        # 生成报告
        self.generate_report()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Checkpoint-669 痤疮诊断模型测试运行器")
    parser.add_argument(
        "--scenario", 
        choices=["smoke_test", "full_regression", "performance_benchmark", "diagnosis_accuracy"],
        default="full_regression",
        help="要运行的测试场景"
    )
    parser.add_argument(
        "--output", 
        choices=["json", "markdown", "html", "both"],
        default="both",
        help="报告输出格式"
    )
    parser.add_argument(
        "--test-dir",
        default="/root/autodl-tmp/test",
        help="测试目录路径"
    )
    
    args = parser.parse_args()
    
    # 检查测试目录
    if not os.path.exists(args.test_dir):
        print(f"错误: 测试目录不存在: {args.test_dir}")
        sys.exit(1)
    
    # 创建测试运行器
    runner = TestRunner(args.test_dir)
    
    # 运行测试
    try:
        runner.run_all_tests(args.scenario)
        print(f"\n测试完成! 报告已保存到: {runner.results_dir}")
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n运行测试时发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 添加缺失的导入
    import importlib.util
    main()