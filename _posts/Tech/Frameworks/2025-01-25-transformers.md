---
title: Huggingface Transformers
date: 2025-01-27
categories: [Tech, Frameworks]
tags: [bert, machine learning, NLP]     # TAG names should always be lowercase
math: true
---

Official Docs: [transformers](https://huggingface.co/docs/transformers/index)

GitHub: [transformers](https://github.com/huggingface/transformers)

**`transformers`** 是 Hugging Face 开发的 **最流行的 NLP & LLM 库**，提供 **预训练 Transformer 模型**，支持 **文本、图像、音频、代码等任务**，并兼容 **PyTorch、TensorFlow 和 JAX**

### Load Pre-trianed Models

加载预训练模型及权重，可用于推理、训练、微调

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 选择模型（例如 LLaMA 2-7B）
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 加载 Tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# 进行推理
prompt = "What is transformers?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

**API**: **Auto Class** 是一组 **自动模型加载器**，用于根据 **模型名称或配置** 自动选择正确的 Transformer 模型、分词器（Tokenizer）或特定任务的 Pipeline。

![auto-class](/assets/images/auto-class.png)

### Pipelines

Pipelines 提供直接面向具体推理任务的API，包括：Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering



```python
from transformers import pipeline

# 文本生成
text_generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
print(text_generator("What is the meaning of life?", max_length=50))

# 文本分类
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("I love this movie!"))

# 翻译
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("Hello, how are you?"))

# 问答
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
context = "Hugging Face is a company that provides NLP technologies and models."
question = "What does Hugging Face provide?"
print(qa_pipeline(question=question, context=context))

# 代码生成
code_generator = pipeline("text-generation", model="bigcode/starcoder")
print(code_generator("def fibonacci(n):", max_length=50))

```

### Trainer + PEFT

Trainer提供了PyTorch训练框架

PEFT：是 **Hugging Face 推出的库**，用于 **高效微调（fine-tuning）大语言模型（LLM）**，减少 **显存占用** 和 **计算成本**，适用于 **LoRA（Low-Rank Adaptation）、Prefix Tuning、Adapter、IA3 等技术**。

PEFT Docs: [PEFT docs](https://huggingface.co/docs/peft/en/index)

Huggingface Transformers可结合peft进行LORA微调：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"

# 加载预训练模型（LoRA 微调前的基模型）
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


from peft import LoraConfig, get_peft_model

# 配置 LoRA 参数
lora_config = LoraConfig(
    r=8,  # 低秩分解维度（减少参数）
    lora_alpha=32,  # LoRA 学习率倍率
    lora_dropout=0.05,  # Dropout 防止过拟合
    target_modules=["q_proj", "v_proj"]  # 只对注意力层进行 LoRA 训练
)

# 将 LoRA 应用于模型
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora_llama2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset  # 你的训练数据
)

trainer.train()
model.save_pretrained("lora_finetuned")


from peft import PeftModel

# 加载 LoRA 微调后的模型
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
lora_model = PeftModel.from_pretrained(base_model, "lora_finetuned")

# 推理
inputs = tokenizer("What is PEFT?", return_tensors="pt").to("cuda")
outputs = lora_model.generate(**inputs)
print(tokenizer.decode(outputs[0]))

```



