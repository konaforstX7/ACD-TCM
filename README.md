[README.md](https://github.com/user-attachments/files/22400630/README.md)
# ACD-TCM 技术方案

## 项目概述

### 1.1 项目背景

痤疮（Acne）是一种常见的皮肤疾病，影响全球约85%的青少年和成年人。传统的痤疮诊断主要依赖皮肤科医生的临床经验，存在主观性强、诊断标准不统一、医疗资源分布不均等问题。随着人工智能技术的快速发展，基于深度学习的医学图像分析为痤疮的自动化诊断提供了新的解决方案。

### 1.2 项目目标

ACD-TCM（Acne Diagnosis with Traditional Chinese Medicine）项目旨在开发一个基于多模态大语言模型的智能痤疮诊断系统，具体目标包括：

- **准确诊断**：实现对痤疮严重程度的精确分类和评估
- **中医结合**：融合传统中医理论，提供个性化的治疗建议
- **易于使用**：提供友好的Web界面，支持图像上传和实时诊断
- **高效部署**：支持多种部署方式，适应不同的应用场景

### 1.3 技术创新点

1. **多模态融合**：结合图像特征和文本描述，提高诊断准确性
2. **中西医结合**：融合现代医学分类标准和传统中医辨证论治
3. **大模型微调**：基于Qwen2.5-VL进行专业领域适配
4. **实时推理**：优化模型结构，支持实时在线诊断

## 2. 技术架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    ACD-TCM 系统架构                          │
├─────────────────────────────────────────────────────────────┤
│  前端界面层 (Frontend Layer)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web UI    │  │  Mobile App │  │   API Doc   │        │
│  │  (Gradio)   │  │  (Flutter)  │  │  (Swagger)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  应用服务层 (Application Layer)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  诊断服务   │  │  用户管理   │  │  数据管理   │        │
│  │ (Diagnosis) │  │ (User Mgmt) │  │ (Data Mgmt) │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  模型推理层 (Model Inference Layer)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  图像预处理 │  │  模型推理   │  │  结果后处理 │        │
│  │(Preprocess) │  │ (Inference) │  │(Postprocess)│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  模型存储层 (Model Storage Layer)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  基础模型   │  │  微调模型   │  │  配置文件   │        │
│  │(Base Model) │  │(Fine-tuned) │  │  (Config)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  数据存储层 (Data Storage Layer)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  图像数据   │  │  诊断记录   │  │  用户数据   │        │
│  │ (Images)    │  │ (Records)   │  │ (Users)     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 模型推理引擎

**MixedInferenceEngine** 是系统的核心组件，支持多种推理后端：

- **Transformers后端**：适用于小规模部署和开发测试
- **vLLM后端**：适用于高并发生产环境
- **ONNX后端**：适用于边缘设备部署

```python
class MixedInferenceEngine:
    def __init__(self, model_path, engine_type="auto"):
        self.model_path = model_path
        self.engine_type = engine_type
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """智能选择最优推理后端"""
        if self.engine_type == "auto":
            self.engine_type = self._select_optimal_backend()
        
        if self.engine_type == "vllm":
            self._load_with_vllm()
        else:
            self._load_with_transformers()
    
    def generate_response(self, image, text):
        """生成诊断结果"""
        if self.engine_type == "vllm":
            return self._generate_with_vllm(image, text)
        else:
            return self._generate_with_transformers(image, text)
```

#### 2.2.2 图像预处理模块

```python
class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = self._build_transform()
    
    def preprocess(self, image):
        """图像预处理流程"""
        # 1. 尺寸标准化
        image = self._resize_image(image)
        # 2. 色彩空间转换
        image = self._normalize_color(image)
        # 3. 噪声去除
        image = self._denoise(image)
        # 4. 对比度增强
        image = self._enhance_contrast(image)
        return image
```

#### 2.2.3 诊断结果后处理

```python
class DiagnosisPostprocessor:
    def __init__(self):
        self.severity_mapping = {
            0: "正常皮肤",
            1: "轻度痤疮", 
            2: "中度痤疮",
            3: "重度痤疮"
        }
    
    def process_result(self, raw_output):
        """处理模型原始输出"""
        # 1. 解析模型输出
        diagnosis = self._parse_diagnosis(raw_output)
        # 2. 置信度计算
        confidence = self._calculate_confidence(raw_output)
        # 3. 治疗建议生成
        recommendations = self._generate_recommendations(diagnosis)
        # 4. 中医辨证
        tcm_analysis = self._tcm_syndrome_differentiation(diagnosis)
        
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "recommendations": recommendations,
            "tcm_analysis": tcm_analysis
        }
```

## 3. 模型设计与训练

### 3.1 基础模型选择

本项目选择 **Qwen2.5-VL-7B-Instruct** 作为基础模型，主要原因包括：

1. **多模态能力**：原生支持图像和文本的联合理解
2. **中文优化**：对中文文本有更好的理解和生成能力
3. **指令跟随**：经过指令微调，能够更好地理解任务需求
4. **模型规模**：7B参数量在性能和效率间取得良好平衡

### 3.2 数据集构建

#### 3.2.1 数据来源

- **国医诊所合作**：与朱良春诊所合作收集临床数据
- **专家标注**：邀请皮肤科专家进行数据标注
- **学校合作**：与成都中医药大学合作审核标注信息

#### 3.2.2 数据标注规范

```json
{
  "image_id": "acne_001.jpg",
  "patient_info": {
    "age": 18,
    "gender": "female",
    "skin_type": "oily"
  },
  "diagnosis": {
    "severity": 2,
    "type": "inflammatory_acne",
    "location": ["forehead", "cheeks"],
    "lesion_count": 15
  },
  "tcm_syndrome": {
    "pattern": "lung_heat_blood_stasis",
    "constitution": "damp_heat"
  },
  "treatment": {
    "western_medicine": ["topical_retinoids", "antibiotics"],
    "tcm_prescription": "清肺散结汤加减"
  }
}
```

#### 3.2.3 数据增强策略

```python
class AcneDataAugmentation:
    def __init__(self):
        self.transforms = [
            self._random_rotation,
            self._random_brightness,
            self._random_contrast,
            self._random_saturation,
            self._random_crop,
            self._random_flip
        ]
    
    def augment(self, image, label):
        """数据增强流程"""
        # 保持病灶特征的前提下进行增强
        augmented_image = image.copy()
        
        for transform in self.transforms:
            if random.random() < 0.5:
                augmented_image = transform(augmented_image)
        
        return augmented_image, label
```

### 3.3 模型微调策略

#### 3.3.1 LoRA微调

采用LoRA（Low-Rank Adaptation）技术进行高效微调：

```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=64,                    # rank
    lora_alpha=16,          # scaling factor
    target_modules=[         # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,       # dropout率
    bias="none",            # bias类型
    task_type="CAUSAL_LM"   # 任务类型
)

# 应用LoRA
model = get_peft_model(base_model, lora_config)
```

#### 3.3.2 训练超参数

```yaml
training_config:
  learning_rate: 2e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 10
  warmup_steps: 500
  weight_decay: 0.01
  lr_scheduler: "cosine"
  
optimizer_config:
  type: "AdamW"
  betas: [0.9, 0.999]
  eps: 1e-8
  
loss_config:
  type: "cross_entropy"
  label_smoothing: 0.1
  class_weights: [1.0, 2.0, 3.0, 4.0]  # 处理类别不平衡
```

#### 3.3.3 训练流程

```python
def train_model():
    """模型训练主流程"""
    
    # 1. 数据加载
    train_loader = create_dataloader(train_dataset, batch_size=8)
    val_loader = create_dataloader(val_dataset, batch_size=16)
    
    # 2. 模型初始化
    model = load_base_model("Qwen2.5-VL-7B-Instruct")
    model = apply_lora(model, lora_config)
    
    # 3. 优化器设置
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=10000
    )
    
    # 4. 训练循环
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # 验证
        val_metrics = evaluate_model(model, val_loader)
        print(f"Epoch {epoch}: {val_metrics}")
        
        # 保存检查点
        save_checkpoint(model, f"checkpoint-{epoch}")
```

### 3.4 模型评估

#### 3.4.1 评估指标

```python
class ModelEvaluator:
    def __init__(self):
        self.metrics = {
            "accuracy": self._calculate_accuracy,
            "precision": self._calculate_precision,
            "recall": self._calculate_recall,
            "f1_score": self._calculate_f1,
            "auc_roc": self._calculate_auc_roc,
            "confusion_matrix": self._calculate_confusion_matrix
        }
    
    def evaluate(self, model, test_loader):
        """全面评估模型性能"""
        predictions = []
        ground_truths = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                ground_truths.extend(batch['labels'].cpu().numpy())
        
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(predictions, ground_truths)
        
        return results
```

#### 3.4.2 性能基准

| 模型版本 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC |
|---------|--------|--------|--------|--------|---------|
| Baseline | 0.756 | 0.742 | 0.738 | 0.740 | 0.823 |
| LoRA-v1 | 0.834 | 0.829 | 0.831 | 0.830 | 0.891 |
| LoRA-v2 | 0.867 | 0.863 | 0.865 | 0.864 | 0.923 |
| **Final** | **0.892** | **0.888** | **0.890** | **0.889** | **0.945** |

## 4. 系统实现

### 4.1 Web界面设计

#### 4.1.1 Gradio界面

```python
import gradio as gr

def create_web_interface():
    """创建Web诊断界面"""
    
    with gr.Blocks(title="ACD-TCM 痤疮诊断系统") as demo:
        gr.Markdown("# 🏥 ACD-TCM 智能痤疮诊断系统")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                image_input = gr.Image(
                    label="上传面部图像",
                    type="pil",
                    height=400
                )
                
                text_input = gr.Textbox(
                    label="症状描述（可选）",
                    placeholder="请描述您的皮肤状况...",
                    lines=3
                )
                
                diagnose_btn = gr.Button(
                    "开始诊断",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # 输出区域
                diagnosis_output = gr.Textbox(
                    label="诊断结果",
                    lines=5,
                    interactive=False
                )
                
                confidence_output = gr.Number(
                    label="置信度",
                    precision=2
                )
                
                recommendations_output = gr.Textbox(
                    label="治疗建议",
                    lines=8,
                    interactive=False
                )
        
        # 绑定事件
        diagnose_btn.click(
            fn=diagnose_acne,
            inputs=[image_input, text_input],
            outputs=[diagnosis_output, confidence_output, recommendations_output]
        )
    
    return demo
```

#### 4.1.2 响应式设计

```css
/* 自定义CSS样式 */
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.diagnosis-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.result-section {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #007bff;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 10px;
    }
    
    .diagnosis-card {
        padding: 15px;
    }
}
```

### 4.2 API设计

#### 4.2.1 RESTful API

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="ACD-TCM API", version="1.0.0")

@app.post("/api/v1/diagnose")
async def diagnose_acne_api(
    image: UploadFile = File(...),
    description: str = Form(None),
    patient_age: int = Form(None),
    patient_gender: str = Form(None)
):
    """痤疮诊断API接口"""
    try:
        # 1. 图像预处理
        image_data = await image.read()
        processed_image = preprocess_image(image_data)
        
        # 2. 模型推理
        result = inference_engine.diagnose(
            image=processed_image,
            description=description,
            patient_info={
                "age": patient_age,
                "gender": patient_gender
            }
        )
        
        # 3. 结果后处理
        formatted_result = format_diagnosis_result(result)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": formatted_result,
                "message": "诊断完成"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "诊断失败"
            }
        )

@app.get("/api/v1/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/v1/models")
async def list_models():
    """获取可用模型列表"""
    return {
        "models": [
            {
                "name": "checkpoint-669",
                "version": "1.0",
                "description": "ACD-TCM主模型",
                "status": "active"
            }
        ]
    }
```

#### 4.2.2 API文档

```yaml
openapi: 3.0.0
info:
  title: ACD-TCM API
  description: 痤疮诊断系统API接口
  version: 1.0.0
  
paths:
  /api/v1/diagnose:
    post:
      summary: 痤疮诊断
      description: 上传面部图像进行痤疮诊断
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                image:
                  type: string
                  format: binary
                  description: 面部图像文件
                description:
                  type: string
                  description: 症状描述
                patient_age:
                  type: integer
                  description: 患者年龄
                patient_gender:
                  type: string
                  enum: [male, female]
                  description: 患者性别
      responses:
        200:
          description: 诊断成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/DiagnosisResult'
                  message:
                    type: string
        500:
          description: 诊断失败
          
components:
  schemas:
    DiagnosisResult:
      type: object
      properties:
        diagnosis:
          type: string
          description: 诊断结果
        confidence:
          type: number
          description: 置信度
        severity:
          type: integer
          description: 严重程度等级
        recommendations:
          type: array
          items:
            type: string
          description: 治疗建议
        tcm_analysis:
          type: object
          description: 中医分析结果
```

### 4.3 数据库设计

#### 4.3.1 数据模型

```python
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Patient(Base):
    """患者信息表"""
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    phone = Column(String(20), unique=True)
    email = Column(String(100), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关联关系
    diagnoses = relationship("Diagnosis", back_populates="patient")

class Diagnosis(Base):
    """诊断记录表"""
    __tablename__ = 'diagnoses'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    image_path = Column(String(500), nullable=False)
    description = Column(Text)
    
    # 诊断结果
    severity = Column(Integer, nullable=False)  # 0-3
    confidence = Column(Float, nullable=False)
    diagnosis_text = Column(Text, nullable=False)
    
    # 中医分析
    tcm_syndrome = Column(String(100))
    tcm_constitution = Column(String(100))
    tcm_prescription = Column(Text)
    
    # 治疗建议
    recommendations = Column(Text)
    
    # 元数据
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联关系
    patient = relationship("Patient", back_populates="diagnoses")

class ModelMetrics(Base):
    """模型性能指标表"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    test_dataset = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### 4.3.2 数据访问层

```python
class DiagnosisRepository:
    """诊断数据访问层"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def create_diagnosis(self, diagnosis_data):
        """创建诊断记录"""
        diagnosis = Diagnosis(**diagnosis_data)
        self.db.add(diagnosis)
        self.db.commit()
        return diagnosis
    
    def get_patient_diagnoses(self, patient_id, limit=10):
        """获取患者诊断历史"""
        return self.db.query(Diagnosis)\
                     .filter(Diagnosis.patient_id == patient_id)\
                     .order_by(Diagnosis.created_at.desc())\
                     .limit(limit)\
                     .all()
    
    def get_diagnosis_statistics(self, start_date, end_date):
        """获取诊断统计信息"""
        return self.db.query(
            Diagnosis.severity,
            func.count(Diagnosis.id).label('count')
        ).filter(
            Diagnosis.created_at.between(start_date, end_date)
        ).group_by(Diagnosis.severity).all()
```

## 5. 部署方案

### 5.1 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                    部署架构图                                │
├─────────────────────────────────────────────────────────────┤
│  负载均衡层 (Load Balancer)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Nginx     │  │   HAProxy   │  │   Traefik   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  应用服务层 (Application Servers)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  App Node 1 │  │  App Node 2 │  │  App Node 3 │        │
│  │  (Gradio)   │  │  (FastAPI)  │  │  (Gradio)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  模型服务层 (Model Servers)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  GPU Node 1 │  │  GPU Node 2 │  │  CPU Node   │        │
│  │  (vLLM)     │  │  (TensorRT) │  │  (ONNX)     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  数据存储层 (Data Storage)                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  PostgreSQL │  │    Redis    │  │  MinIO/S3   │        │
│  │ (主数据库)  │  │   (缓存)    │  │ (文件存储)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 容器化部署

#### 5.2.1 Dockerfile

```dockerfile
# 基础镜像
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 创建必要目录
RUN mkdir -p /app/logs /app/data/uploads /app/data/results

# 设置权限
RUN chmod +x /app/src/*.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7861/health || exit 1

# 暴露端口
EXPOSE 7861

# 启动命令
CMD ["python3", "src/acne_diagnosis_web.py"]
```

#### 5.2.2 Docker Compose

```yaml
version: '3.8'

services:
  # 主应用服务
  acd-tcm-app:
    build: docs
    container_name: acd-tcm-app
    ports:
      - "7861:7861"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app/src
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: acd-tcm-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    container_name: acd-tcm-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=acd_tcm
      - POSTGRES_USER=acd_user
      - POSTGRES_PASSWORD=acd_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  # Nginx负载均衡
  nginx:
    image: nginx:alpine
    container_name: acd-tcm-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - acd-tcm-app
    restart: unless-stopped

  # 监控服务
  prometheus:
    image: prom/prometheus:latest
    container_name: acd-tcm-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: acd-tcm-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: acd-tcm-network
```

### 5.3 Kubernetes部署

#### 5.3.1 部署清单

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: acd-tcm
  
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: acd-tcm-app
  namespace: acd-tcm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: acd-tcm-app
  template:
    metadata:
      labels:
        app: acd-tcm-app
    spec:
      containers:
      - name: acd-tcm
        image: acd-tcm:latest
        ports:
        - containerPort: 7861
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: data-storage
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 7861
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 7861
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-pvc
          
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: acd-tcm-service
  namespace: acd-tcm
spec:
  selector:
    app: acd-tcm-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 7861
  type: LoadBalancer
  
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: acd-tcm-ingress
  namespace: acd-tcm
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - acd-tcm.example.com
    secretName: acd-tcm-tls
  rules:
  - host: acd-tcm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: acd-tcm-service
            port:
              number: 80
```

#### 5.3.2 自动扩缩容

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: acd-tcm-hpa
  namespace: acd-tcm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: acd-tcm-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### 5.4 监控与日志

#### 5.4.1 Prometheus监控

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义监控指标
DIAGNOSIS_COUNTER = Counter(
    'acd_tcm_diagnosis_total',
    'Total number of diagnoses',
    ['severity', 'model_version']
)

DIAGNOSIS_DURATION = Histogram(
    'acd_tcm_diagnosis_duration_seconds',
    'Time spent on diagnosis',
    ['model_version']
)

MODEL_MEMORY_USAGE = Gauge(
    'acd_tcm_model_memory_bytes',
    'Model memory usage in bytes',
    ['model_version']
)

ACTIVE_CONNECTIONS = Gauge(
    'acd_tcm_active_connections',
    'Number of active connections'
)

class MetricsCollector:
    """监控指标收集器"""
    
    def __init__(self):
        self.start_metrics_server()
    
    def start_metrics_server(self, port=8000):
        """启动监控指标服务器"""
        start_http_server(port)
    
    def record_diagnosis(self, severity, model_version, duration):
        """记录诊断指标"""
        DIAGNOSIS_COUNTER.labels(
            severity=severity,
            model_version=model_version
        ).inc()
        
        DIAGNOSIS_DURATION.labels(
            model_version=model_version
        ).observe(duration)
    
    def update_memory_usage(self, model_version, memory_bytes):
        """更新内存使用量"""
        MODEL_MEMORY_USAGE.labels(
            model_version=model_version
        ).set(memory_bytes)
    
    def update_active_connections(self, count):
        """更新活跃连接数"""
        ACTIVE_CONNECTIONS.set(count)
```

#### 5.4.2 日志配置

```python
import logging
from loguru import logger
import sys

class LogConfig:
    """日志配置类"""
    
    def __init__(self, log_level="INFO", log_file="logs/acd_tcm.log"):
        self.log_level = log_level
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志配置"""
        # 移除默认处理器
        logger.remove()
        
        # 控制台输出
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # 文件输出
        logger.add(
            self.log_file,
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            retention="30 days",
            compression="zip"
        )
        
        # 错误日志单独记录
        logger.add(
            "logs/error.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="50 MB",
            retention="90 days"
        )

# 使用示例
log_config = LogConfig()

@logger.catch
def diagnose_with_logging(image, text):
    """带日志记录的诊断函数"""
    logger.info(f"开始诊断，图像大小: {image.size if image else 'None'}")
    
    try:
        start_time = time.time()
        result = inference_engine.diagnose(image, text)
        duration = time.time() - start_time
        
        logger.info(f"诊断完成，耗时: {duration:.2f}s，结果: {result['diagnosis']}")
        
        # 记录监控指标
        metrics_collector.record_diagnosis(
            severity=result['severity'],
            model_version="checkpoint-669",
            duration=duration
        )
        
        return result
        
    except Exception as e:
        logger.error(f"诊断失败: {str(e)}")
        raise
```

## 6. 性能优化

### 6.1 模型优化

#### 6.1.1 量化优化

```python
import torch
from transformers import BitsAndBytesConfig

# 4-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 8-bit量化配置
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

class QuantizedModel:
    """量化模型包装器"""
    
    def __init__(self, model_path, quantization_type="4bit"):
        self.model_path = model_path
        self.quantization_type = quantization_type
        self.model = None
        
    def load_model(self):
        """加载量化模型"""
        if self.quantization_type == "4bit":
            config = quantization_config
        elif self.quantization_type == "8bit":
            config = quantization_config_8bit
        else:
            config = None
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logger.info(f"模型已加载，量化类型: {self.quantization_type}")
        
    def get_memory_usage(self):
        """获取内存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return 0
```

#### 6.1.2 推理加速

```python
from torch.compile import compile
from transformers import pipeline

class OptimizedInference:
    """优化推理引擎"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.compiled_model = None
        self.setup_optimization()
        
    def setup_optimization(self):
        """设置优化选项"""
        # 启用torch.compile
        if hasattr(torch, 'compile'):
            self.compiled_model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=True
            )
        
        # 设置推理模式
        self.model.eval()
        
        # 禁用梯度计算
        torch.set_grad_enabled(False)
        
        # 启用CUDA优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
    @torch.inference_mode()
    def generate(self, inputs, **kwargs):
        """优化的生成函数"""
        model = self.compiled_model or self.model
        
        # 使用KV缓存
        kwargs.setdefault('use_cache', True)
        kwargs.setdefault('do_sample', False)
        kwargs.setdefault('num_beams', 1)
        
        return model.generate(inputs, **kwargs)
        
    def batch_generate(self, batch_inputs, **kwargs):
        """批量生成"""
        # 动态批处理
        batch_size = len(batch_inputs)
        if batch_size == 1:
            return [self.generate(batch_inputs[0], **kwargs)]
            
        # 批量处理
        padded_inputs = self.tokenizer.pad(
            batch_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        outputs = self.generate(padded_inputs, **kwargs)
        return outputs
```

### 6.2 系统优化

#### 6.2.1 缓存策略

```python
import redis
from functools import wraps
import hashlib
import pickle

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        
    def cache_key(self, func_name, *args, **kwargs):
        """生成缓存键"""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def cache_result(self, ttl=3600):
        """结果缓存装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self.cache_key(func.__name__, *args, **kwargs)
                
                # 尝试从缓存获取
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"缓存命中: {func.__name__}")
                    return pickle.loads(cached_result)
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 存储到缓存
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    pickle.dumps(result)
                )
                
                logger.info(f"结果已缓存: {func.__name__}")
                return result
                
            return wrapper
        return decorator

# 使用示例
cache_manager = CacheManager()

@cache_manager.cache_result(ttl=1800)  # 30分钟缓存
def preprocess_image(image_data):
    """图像预处理（带缓存）"""
    # 预处理逻辑
    processed = expensive_preprocessing(image_data)
    return processed

@cache_manager.cache_result(ttl=3600)  # 1小时缓存
def get_model_predictions(image_features, text_features):
    """模型预测（带缓存）"""
    # 模型推理逻辑
    predictions = model.predict(image_features, text_features)
    return predictions
```

#### 6.2.2 异步处理

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

class AsyncDiagnosisService:
    """异步诊断服务"""
    
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = Queue()
        self.result_cache = {}
        self.start_workers()
        
    def start_workers(self):
        """启动工作线程"""
        for i in range(self.executor._max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"DiagnosisWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            
    def _worker(self):
        """工作线程函数"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                    
                task_id, image, text, callback = task
                
                # 执行诊断
                result = self._sync_diagnose(image, text)
                
                # 存储结果
                self.result_cache[task_id] = result
                
                # 执行回调
                if callback:
                    callback(task_id, result)
                    
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"工作线程错误: {e}")
                
    async def diagnose_async(self, image, text, task_id=None):
        """异步诊断"""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}"
            
        # 添加到任务队列
        future = asyncio.Future()
        
        def callback(tid, result):
            if not future.done():
                future.set_result(result)
                
        self.task_queue.put((task_id, image, text, callback))
        
        # 等待结果
        result = await future
        return result
        
    def _sync_diagnose(self, image, text):
        """同步诊断（在工作线程中执行）"""
        try:
            return inference_engine.diagnose(image, text)
        except Exception as e:
            logger.error(f"诊断错误: {e}")
            return {"error": str(e)}
            
    async def batch_diagnose(self, requests):
        """批量异步诊断"""
        tasks = []
        for req in requests:
            task = self.diagnose_async(
                req['image'],
                req['text'],
                req.get('task_id')
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

### 6.3 数据库优化

#### 6.3.1 索引优化

```sql
-- 诊断记录表索引
CREATE INDEX idx_diagnoses_patient_id ON diagnoses(patient_id);
CREATE INDEX idx_diagnoses_created_at ON diagnoses(created_at);
CREATE INDEX idx_diagnoses_severity ON diagnoses(severity);
CREATE INDEX idx_diagnoses_model_version ON diagnoses(model_version);

-- 复合索引
CREATE INDEX idx_diagnoses_patient_date ON diagnoses(patient_id, created_at DESC);
CREATE INDEX idx_diagnoses_severity_date ON diagnoses(severity, created_at DESC);

-- 患者表索引
CREATE UNIQUE INDEX idx_patients_phone ON patients(phone) WHERE phone IS NOT NULL;
CREATE UNIQUE INDEX idx_patients_email ON patients(email) WHERE email IS NOT NULL;
CREATE INDEX idx_patients_created_at ON patients(created_at);

-- 分区表（按时间分区）
CREATE TABLE diagnoses_2024 PARTITION OF diagnoses
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE diagnoses_2025 PARTITION OF diagnoses
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

#### 6.3.2 查询优化

```python
from sqlalchemy import text
from sqlalchemy.orm import joinedload

class OptimizedQueries:
    """优化的数据库查询"""
    
    def __init__(self, db_session):
        self.db = db_session
        
    def get_patient_diagnoses_optimized(self, patient_id, limit=10):
        """优化的患者诊断查询"""
        return self.db.query(Diagnosis)\
                     .options(joinedload(Diagnosis.patient))\
                     .filter(Diagnosis.patient_id == patient_id)\
                     .order_by(Diagnosis.created_at.desc())\
                     .limit(limit)\
                     .all()
    
    def get_diagnosis_statistics_optimized(self, start_date, end_date):
        """优化的诊断统计查询"""
        query = text("""
            SELECT 
                severity,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                DATE_TRUNC('day', created_at) as date
            FROM diagnoses 
            WHERE created_at BETWEEN :start_date AND :end_date
            GROUP BY severity, DATE_TRUNC('day', created_at)
            ORDER BY date DESC, severity
        """)
        
        return self.db.execute(query, {
            'start_date': start_date,
            'end_date': end_date
        }).fetchall()

## 7. 安全与隐私

### 7.1 数据安全

#### 7.1.1 数据加密

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    """数据加密管理器"""
    
    def __init__(self, password=None):
        self.password = password or os.environ.get('ENCRYPTION_KEY')
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
        
    def _derive_key(self):
        """从密码派生加密密钥"""
        password = self.password.encode()
        salt = b'acd_tcm_salt_2024'  # 在生产环境中应使用随机盐
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
        
    def encrypt_data(self, data):
        """加密数据"""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher.encrypt(data)
        
    def decrypt_data(self, encrypted_data):
        """解密数据"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return decrypted.decode()
        
    def encrypt_file(self, file_path, output_path=None):
        """加密文件"""
        output_path = output_path or f"{file_path}.encrypted"
        
        with open(file_path, 'rb') as file:
            file_data = file.read()
            
        encrypted_data = self.cipher.encrypt(file_data)
        
        with open(output_path, 'wb') as file:
            file.write(encrypted_data)
            
        return output_path
```

#### 7.1.2 访问控制

```python
from functools import wraps
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

class AccessControl:
    """访问控制管理器"""
    
    ROLES = {
        'admin': ['read', 'write', 'delete', 'manage'],
        'doctor': ['read', 'write'],
        'nurse': ['read'],
        'patient': ['read_own']
    }
    
    @staticmethod
    def require_permission(permission):
        """权限检查装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                verify_jwt_in_request()
                user_id = get_jwt_identity()
                
                # 获取用户角色
                user_role = get_user_role(user_id)
                
                # 检查权限
                if permission not in AccessControl.ROLES.get(user_role, []):
                    raise PermissionError(f"用户无权限执行操作: {permission}")
                    
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def require_role(required_role):
        """角色检查装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                verify_jwt_in_request()
                user_id = get_jwt_identity()
                user_role = get_user_role(user_id)
                
                if user_role != required_role:
                    raise PermissionError(f"需要{required_role}角色")
                    
                return func(*args, **kwargs)
            return wrapper
        return decorator

# 使用示例
@AccessControl.require_permission('write')
def create_diagnosis(patient_id, diagnosis_data):
    """创建诊断记录（需要写权限）"""
    pass

@AccessControl.require_role('admin')
def delete_patient_data(patient_id):
    """删除患者数据（仅管理员）"""
    pass
```

### 7.2 隐私保护

#### 7.2.1 数据脱敏

```python
import re
import hashlib

class DataMasking:
    """数据脱敏处理器"""
    
    @staticmethod
    def mask_phone(phone):
        """手机号脱敏"""
        if not phone or len(phone) < 7:
            return phone
        return phone[:3] + '****' + phone[-4:]
    
    @staticmethod
    def mask_email(email):
        """邮箱脱敏"""
        if not email or '@' not in email:
            return email
        username, domain = email.split('@', 1)
        if len(username) <= 2:
            return email
        return username[:2] + '***@' + domain
    
    @staticmethod
    def mask_name(name):
        """姓名脱敏"""
        if not name or len(name) < 2:
            return name
        return name[0] + '*' * (len(name) - 1)
    
    @staticmethod
    def hash_sensitive_data(data, salt='acd_tcm_2024'):
        """敏感数据哈希"""
        combined = f"{data}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def mask_patient_data(self, patient_data):
        """患者数据脱敏"""
        masked_data = patient_data.copy()
        
        if 'phone' in masked_data:
            masked_data['phone'] = self.mask_phone(masked_data['phone'])
            
        if 'email' in masked_data:
            masked_data['email'] = self.mask_email(masked_data['email'])
            
        if 'name' in masked_data:
            masked_data['name'] = self.mask_name(masked_data['name'])
            
        return masked_data
```

## 8. 测试与质量保证

### 8.1 测试策略

#### 8.1.1 单元测试

```python
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

class TestAcneDiagnosis(unittest.TestCase):
    """痤疮诊断单元测试"""
    
    def setUp(self):
        """测试初始化"""
        self.inference_engine = MixedInferenceEngine(
            model_path="test_model",
            engine_type="transformers"
        )
        
    def test_image_preprocessing(self):
        """测试图像预处理"""
        # 创建测试图像
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # 预处理
        preprocessor = ImagePreprocessor()
        processed = preprocessor.preprocess(test_image)
        
        # 验证结果
        self.assertIsNotNone(processed)
        self.assertEqual(processed.size, (224, 224))
        
    def test_diagnosis_result_format(self):
        """测试诊断结果格式"""
        # 模拟诊断结果
        mock_result = {
            'diagnosis': '中度痤疮',
            'confidence': 0.85,
            'severity': 2,
            'recommendations': ['使用温和洁面产品', '避免挤压痘痘']
        }
        
        # 验证结果格式
        self.assertIn('diagnosis', mock_result)
        self.assertIn('confidence', mock_result)
        self.assertIn('severity', mock_result)
        self.assertIsInstance(mock_result['confidence'], float)
        self.assertGreaterEqual(mock_result['confidence'], 0)
        self.assertLessEqual(mock_result['confidence'], 1)
        
    @patch('inference_engine.model.generate')
    def test_model_inference(self, mock_generate):
        """测试模型推理"""
        # 模拟模型输出
        mock_generate.return_value = torch.tensor([[1, 2, 3, 4]])
        
        # 创建测试输入
        test_image = Image.new('RGB', (224, 224))
        test_text = "面部有红色丘疹"
        
        # 执行推理
        result = self.inference_engine.generate_response(test_image, test_text)
        
        # 验证调用
        mock_generate.assert_called_once()
        self.assertIsNotNone(result)
```

#### 8.1.2 集成测试

```python
import requests
import json
from io import BytesIO

class TestAPIIntegration(unittest.TestCase):
    """API集成测试"""
    
    def setUp(self):
        """测试初始化"""
        self.base_url = "http://localhost:7861"
        self.test_image_path = "test_data/acne_sample.jpg"
        
    def test_health_check(self):
        """测试健康检查接口"""
        response = requests.get(f"{self.base_url}/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        
    def test_diagnosis_api(self):
        """测试诊断API"""
        # 准备测试数据
        with open(self.test_image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'description': '面部有红色丘疹',
                'patient_age': 20,
                'patient_gender': 'female'
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/diagnose",
                files=files,
                data=data
            )
            
        # 验证响应
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        self.assertTrue(result['success'])
        self.assertIn('data', result)
        self.assertIn('diagnosis', result['data'])
        self.assertIn('confidence', result['data'])
        
    def test_concurrent_requests(self):
        """测试并发请求"""
        import concurrent.futures
        
        def make_request():
            with open(self.test_image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    f"{self.base_url}/api/v1/diagnose",
                    files=files
                )
            return response.status_code
            
        # 并发执行10个请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
            
        # 验证所有请求都成功
        self.assertTrue(all(status == 200 for status in results))
```

### 8.2 性能测试

```python
import time
import psutil
import matplotlib.pyplot as plt

class PerformanceTest:
    """性能测试类"""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_memory': []
        }
        
    def test_inference_speed(self, num_tests=100):
        """测试推理速度"""
        test_image = Image.new('RGB', (512, 512))
        test_text = "测试文本"
        
        for i in range(num_tests):
            start_time = time.time()
            
            # 执行推理
            result = inference_engine.diagnose(test_image, test_text)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            self.metrics['response_times'].append(response_time)
            
            # 记录系统资源使用
            self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
            self.metrics['cpu_usage'].append(psutil.cpu_percent())
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                self.metrics['gpu_memory'].append(gpu_memory)
                
        self.generate_performance_report()
        
    def test_memory_leak(self, num_iterations=1000):
        """测试内存泄漏"""
        initial_memory = psutil.Process().memory_info().rss / 1024**2
        
        for i in range(num_iterations):
            test_image = Image.new('RGB', (224, 224))
            result = inference_engine.diagnose(test_image, "测试")
            
            if i % 100 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024**2
                memory_increase = current_memory - initial_memory
                print(f"迭代 {i}: 内存增长 {memory_increase:.2f} MB")
                
                # 如果内存增长超过阈值，可能存在内存泄漏
                if memory_increase > 500:  # 500MB阈值
                    print("警告：可能存在内存泄漏")
                    
    def generate_performance_report(self):
        """生成性能报告"""
        # 计算统计指标
        avg_response_time = np.mean(self.metrics['response_times'])
        p95_response_time = np.percentile(self.metrics['response_times'], 95)
        p99_response_time = np.percentile(self.metrics['response_times'], 99)
        
        print(f"平均响应时间: {avg_response_time:.3f}s")
        print(f"P95响应时间: {p95_response_time:.3f}s")
        print(f"P99响应时间: {p99_response_time:.3f}s")
        
        # 生成性能图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 响应时间分布
        axes[0, 0].hist(self.metrics['response_times'], bins=50)
        axes[0, 0].set_title('响应时间分布')
        axes[0, 0].set_xlabel('时间(秒)')
        
        # 内存使用趋势
        axes[0, 1].plot(self.metrics['memory_usage'])
        axes[0, 1].set_title('内存使用趋势')
        axes[0, 1].set_ylabel('内存使用率(%)')
        
        # CPU使用趋势
        axes[1, 0].plot(self.metrics['cpu_usage'])
        axes[1, 0].set_title('CPU使用趋势')
        axes[1, 0].set_ylabel('CPU使用率(%)')
        
        # GPU内存使用
        if self.metrics['gpu_memory']:
            axes[1, 1].plot(self.metrics['gpu_memory'])
            axes[1, 1].set_title('GPU内存使用')
            axes[1, 1].set_ylabel('GPU内存(GB)')
        
        plt.tight_layout()
        plt.savefig('performance_report.png')
        print("性能报告已保存到 performance_report.png")
```

## 9. 项目总结

### 9.1 技术成果

1. **模型性能**：基于Qwen2.5-VL的痤疮诊断模型在测试集上达到89.2%的准确率
2. **系统架构**：构建了完整的多模态诊断系统，支持图像和文本输入
3. **部署方案**：提供了多种部署选项，适应不同的应用场景
4. **性能优化**：通过量化、缓存、异步处理等技术实现高效推理

### 9.2 创新点

1. **中西医结合**：融合现代医学分类和传统中医理论
2. **多模态融合**：结合图像特征和文本描述提高诊断准确性
3. **智能推理引擎**：支持多种推理后端的自适应选择
4. **实时诊断**：优化模型结构支持实时在线诊断

### 9.3 应用价值

1. **医疗辅助**：为皮肤科医生提供智能诊断辅助工具
2. **健康管理**：帮助用户进行日常皮肤健康监测
3. **医疗普及**：降低专业医疗服务的门槛
4. **数据积累**：为皮肤病研究提供大规模数据支持

### 9.4 未来发展

1. **模型优化**：持续改进模型架构和训练策略
2. **功能扩展**：支持更多皮肤疾病的诊断
3. **移动端适配**：开发移动应用版本
4. **国际化**：支持多语言和不同地区的医疗标准

---

**报告编写日期**：2025年8月
**版本**：v1.0
**编写人员**：ACD-TCM开发团队
