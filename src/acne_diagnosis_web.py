import copy
import re
import os
import glob
from argparse import ArgumentParser
from threading import Thread
import gc

import gradio as gr
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer
from peft import PeftModel

DEFAULT_CKPT_PATH = '/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct'
TRAINING_OUTPUT_DIR = '/root/autodl-tmp/Qwen/Qwen2.5-VL/output/Qwen2_5-VL-7B-Acne'

# 痤疮诊断专用prompt模板
ACNE_DIAGNOSIS_PROMPT = """你是一位专业的中医皮肤科医生，擅长痤疮的诊断和治疗。请根据提供的面部图像，进行详细的痤疮分析和诊断。

请按照以下格式进行分析：

1. **痤疮类型识别**：
   - 识别图像中的痤疮类型（如：粉刺、丘疹、脓疱、结节、囊肿等）
   - 描述痤疮的分布位置和严重程度

2. **中医辨证分析**：
   - 根据痤疮的表现特征进行中医证型分析
   - 可能的证型包括：肺经风热、脾胃湿热、冲任不调、痰湿凝结、血瘀痰结等

3. **诊断结论**：
   - 给出明确的痤疮诊断
   - 评估严重程度（轻度、中度、重度）

4. **治疗建议**：
   - 中医治疗方案（如中药方剂、针灸等）
   - 日常护理建议
   - 饮食调理建议

请基于图像内容进行专业、详细的分析。"""

def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Base model checkpoint path')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')
    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Demo server name.')
    args = parser.parse_args()
    return args

def get_available_checkpoints():
    """获取可用的checkpoint列表"""
    checkpoints = []
    
    # 添加基础模型
    checkpoints.append(("基础模型 (Base Model)", DEFAULT_CKPT_PATH))
    
    # 查找训练输出目录中的checkpoint
    if os.path.exists(TRAINING_OUTPUT_DIR):
        checkpoint_dirs = glob.glob(os.path.join(TRAINING_OUTPUT_DIR, "checkpoint-*"))
        checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
        
        for checkpoint_dir in checkpoint_dirs:
            step_num = checkpoint_dir.split('-')[-1]
            display_name = f"微调模型 Step {step_num}"
            checkpoints.append((display_name, checkpoint_dir))
    
    return checkpoints

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def _load_model_processor(base_model_path, checkpoint_path=None, cpu_only=False, flash_attn2=False):
    """加载模型和处理器"""
    # 清理内存
    clear_gpu_memory()
    
    if cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    # 加载基础模型，使用内存优化选项
    if flash_attn2:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # 使用半精度减少内存占用
            attn_implementation='flash_attention_2',
            device_map=device_map,
            low_cpu_mem_usage=True     # 减少CPU内存使用
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, 
            device_map=device_map,
            torch_dtype=torch.float16,  # 使用半精度减少内存占用
            low_cpu_mem_usage=True     # 减少CPU内存使用
        )

    # 如果指定了checkpoint路径且不是基础模型，则加载LoRA权重
    if checkpoint_path and checkpoint_path != base_model_path:
        print(f"Loading LoRA weights from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()  # 合并LoRA权重

    processor = AutoProcessor.from_pretrained(base_model_path)
    return model, processor

def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)

def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages

def _launch_demo(args):
    # 获取可用的checkpoints
    available_checkpoints = get_available_checkpoints()
    
    # 全局变量存储当前模型
    current_model = None
    current_processor = None
    current_checkpoint = None

    def load_checkpoint(checkpoint_choice):
        nonlocal current_model, current_processor, current_checkpoint
        
        # 找到选择的checkpoint路径
        selected_checkpoint = None
        for display_name, checkpoint_path in available_checkpoints:
            if display_name == checkpoint_choice:
                selected_checkpoint = checkpoint_path
                break
        
        if selected_checkpoint is None:
            return "❌ 未找到选择的checkpoint"
        
        if current_checkpoint == selected_checkpoint:
            return f"✅ 当前已加载: {checkpoint_choice}"
        
        try:
            # 清理之前的模型和GPU内存
            if current_model is not None:
                del current_model
                del current_processor
                _gc()
                clear_gpu_memory()
            
            # 加载新模型
            current_model, current_processor = _load_model_processor(
                args.checkpoint_path,
                selected_checkpoint,
                args.cpu_only,
                args.flash_attn2
            )
            current_checkpoint = selected_checkpoint
            
            return f"✅ 成功加载: {checkpoint_choice}"
        except torch.cuda.OutOfMemoryError as e:
            clear_gpu_memory()
            return f"❌ GPU内存不足，无法加载模型。请尝试：\n1. 重启程序\n2. 关闭其他占用GPU的程序\n错误详情: {str(e)}"
        except Exception as e:
            clear_gpu_memory()
            return f"❌ 加载失败: {str(e)}"

    def call_local_model(messages):
        if current_model is None or current_processor is None:
            yield "请先选择并加载一个checkpoint模型"
            return

        try:
            # 清理GPU内存
            clear_gpu_memory()
            
            messages = _transform_messages(messages)
            
            # 使用torch.no_grad()减少内存占用
            with torch.no_grad():
                text = current_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = current_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
                inputs = inputs.to(current_model.device)

                tokenizer = current_processor.tokenizer
                streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

                # 减少max_new_tokens以节省内存
                gen_kwargs = {
                    'max_new_tokens': 512,  # 减少生成长度以节省内存
                    'streamer': streamer, 
                    'pad_token_id': tokenizer.eos_token_id,
                    **inputs
                }

                thread = Thread(target=current_model.generate, kwargs=gen_kwargs)
                thread.start()

                generated_text = ''
                for new_text in streamer:
                    generated_text += new_text
                    yield generated_text
                    
            # 清理内存
            del inputs
            clear_gpu_memory()
            
        except torch.cuda.OutOfMemoryError as e:
            clear_gpu_memory()
            yield f"GPU内存不足，请尝试以下解决方案：\n1. 重启Web界面\n2. 使用更小的图片\n3. 关闭其他占用GPU的程序\n错误详情: {str(e)}"
        except Exception as e:
            clear_gpu_memory()
            yield f"生成失败: {str(e)}"

    def predict_acne(_chatbot, task_history, use_template):
        if not task_history:
            return _chatbot
        
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        
        if len(chat_query) == 0:
            _chatbot.pop()
            task_history.pop()
            return _chatbot
        
        print('User: ' + _parse_text(str(query)))
        
        # 构建消息
        messages = []
        content = []
        
        # 添加图像
        if isinstance(query, (tuple, list)):
            content.append({'image': f'file://{query[0]}'})
        
        # 添加文本prompt
        if use_template:
            content.append({'text': ACNE_DIAGNOSIS_PROMPT})
        else:
            # 如果不使用模板，添加简单的诊断请求
            content.append({'text': '请分析这张面部图像中的痤疮情况。'})
        
        messages.append({'role': 'user', 'content': content})
        
        # 生成回复
        for response in call_local_model(messages):
            _chatbot[-1] = (_parse_text(str(chat_query)), _remove_image_special(_parse_text(response)))
            yield _chatbot
        
        task_history[-1] = (query, _parse_text(response))
        print('Acne Diagnosis: ' + _parse_text(response))
        yield _chatbot

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_state(_chatbot, task_history):
        task_history.clear()
        _chatbot.clear()
        _gc()
        return []

    # 创建Gradio界面
    with gr.Blocks(title="痤疮诊断AI助手") as demo:
        gr.Markdown("""
        # 🏥 痤疮诊断AI助手
        
        基于Qwen2.5-VL的中医痤疮诊断系统，支持多种微调模型选择。
        
        ## 使用说明：
        1. 选择要使用的模型checkpoint
        2. 上传面部痤疮图像
        3. 选择是否使用专业诊断模板
        4. 点击诊断按钮获取分析结果
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Checkpoint选择
                checkpoint_choices = [name for name, _ in available_checkpoints]
                checkpoint_dropdown = gr.Dropdown(
                    choices=checkpoint_choices,
                    label="选择模型 Checkpoint",
                    value=checkpoint_choices[0] if checkpoint_choices else None
                )
                
                load_btn = gr.Button("🔄 加载模型", variant="primary")
                load_status = gr.Textbox(label="加载状态", interactive=False)
                
                # 诊断选项
                use_template = gr.Checkbox(
                    label="使用专业诊断模板",
                    value=True,
                    info="启用后将使用详细的中医诊断prompt"
                )
            
            with gr.Column(scale=2):
                # 聊天界面
                chatbot = gr.Chatbot(
                    label='痤疮诊断结果',
                    elem_classes='control-height',
                    height=400
                )
                
                task_history = gr.State([])
                
                with gr.Row():
                    upload_btn = gr.UploadButton(
                        "📷 上传面部图像",
                        file_types=['image'],
                        variant="secondary"
                    )
                    diagnose_btn = gr.Button("🔍 开始诊断", variant="primary")
                    clear_btn = gr.Button("🧹 清除历史", variant="secondary")
        
        # 事件绑定
        load_btn.click(
            load_checkpoint,
            inputs=[checkpoint_dropdown],
            outputs=[load_status]
        )
        
        upload_btn.upload(
            add_file,
            inputs=[chatbot, task_history, upload_btn],
            outputs=[chatbot, task_history]
        )
        
        diagnose_btn.click(
            predict_acne,
            inputs=[chatbot, task_history, use_template],
            outputs=[chatbot],
            show_progress=True
        )
        
        clear_btn.click(
            reset_state,
            inputs=[chatbot, task_history],
            outputs=[chatbot]
        )
        
        gr.Markdown("""
        ---
        
        **注意事项：**
        - 本系统的结果仅供参考，不能替代专业医疗诊断
        - 请上传清晰的面部图像以获得更准确的分析
        - 建议结合专业医生的意见进行综合判断
        
        
        """)
    
    # 启动demo
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )

def main():
    args = _get_args()
    _launch_demo(args)

if __name__ == '__main__':
    main()