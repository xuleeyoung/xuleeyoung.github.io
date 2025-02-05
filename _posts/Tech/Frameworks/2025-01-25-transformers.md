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



### Transformers Source Code Structure

The `transformers` library consists of the following major components:

| Component           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **`models/`**       | Implements different Transformer architectures (e.g., BERT, GPT, T5, LLaMA) |
| **`tokenization/`** | Handles text tokenization and special tokens (e.g., `[CLS]`, `[SEP]`) |
| **`modeling/`**     | Contains the Transformer models (PyTorch & TensorFlow implementations) |
| **`trainer/`**      | Handles training & fine-tuning using `Trainer` API           |
| **`generation/`**   | Implements text generation algorithms (beam search, sampling, etc.) |
| **`pipelines/`**    | Provides easy-to-use APIs for inference (e.g., `pipeline("text-generation")`) |

The `transformers` library is built around **four main classes**:

1. **`PreTrainedModel`** (Model Class) → Loads pre-trained weights and defines architectures
2. **`PreTrainedTokenizer`** (Tokenizer Class) → Handles tokenization, padding, and special tokens
3. **`Trainer`** (Training Class) → Simplifies training and fine-tuning
4. **`Pipeline`** (Inference API) → High-level API for model inference

```sql
+---------------------+
| PreTrainedModel    | <----- Defines Model Architecture (BERT, GPT, T5)
+---------------------+
        |
        v
+---------------------+
| AutoModel          | <----- Loads Any Pre-Trained Model
+---------------------+
        |
        v
+---------------------+
| Specific Models    | <----- e.g., BertModel, GPT2Model, T5ForConditionalGeneration
+---------------------+

+----------------------+
| PreTrainedTokenizer | <----- Handles tokenization
+----------------------+
        |
        v
+----------------------+
| AutoTokenizer       | <----- Loads Tokenizer for Any Model
+----------------------+
        |
        v
+----------------------+
| Specific Tokenizers | <----- e.g., BertTokenizer, GPT2Tokenizer
+----------------------+

+-------------------+
| Trainer          | <----- High-Level Training API
+-------------------+

+-------------------+
| Pipeline         | <----- High-Level Inference API
+-------------------+

```

**Key Classes**:

###### Model Class(PreTrainedModel):

- This is the **base class** for all Transformer models.

- It defines methods for **loading, saving, and configuring models**.

- Every specific model (like `BertModel`, `GPT2Model`) **inherits from this**.

 **Key Model Variants**:

| Class                           | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| `BertModel`                     | Basic BERT encoder (without classification head) |
| `BertForSequenceClassification` | BERT with a classification head                  |
| `GPT2Model`                     | GPT-2 decoder (without LM head)                  |
| `GPT2LMHeadModel`               | GPT-2 with a language modeling head              |
| `T5ForConditionalGeneration`    | T5 model for text-to-text generation             |

###### Tokenizer Class(PreTrainedTokenizer):

- Handles **text tokenization** for transformer models.

- Converts text into token IDs and manages **padding, truncation, and special tokens**.

- Each model has its own tokenizer (`BertTokenizer`, `GPT2Tokenizer`, etc.).



###### Trainer

###### Pipeline

#### Case Study: Parameter/API of GPT2Model

class transformers.GPT2Model: 

```python
def forward(
	self,
  input_ids: Optional[torch.LongTensor] = None,
  past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
  attention_mask: Optional[torch.FloatTensor] = None,
  token_type_ids: Optional[torch.LongTensor] = None,
  position_ids: Optional[torch.LongTensor] = None,
  head_mask: Optional[torch.FloatTensor] = None,
  inputs_embeds: Optional[torch.FloatTensor] = None,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  encoder_attention_mask: Optional[torch.FloatTensor] = None,
  use_cache: Optional[bool] = None,
  output_attentions: Optional[bool] = None,
  output_hidden_states: Optional[bool] = None,
  return_dict: Optional[bool] = None,
)
```

output of forward has the fields:

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.

- **past_key_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

class transformers.GPT2LMHeadModel:

The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

forward method has the same input types as GPT2Model, with a plus of `labels`.

output has the same fields as GPT2Model and additionally:

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).







