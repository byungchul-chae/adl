from huggingface_hub import login

# Hugging Face에 로그인
login(token="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")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

# llama2 모델과 토크나이저 불러오기
# GPT-2 모델과 토크나이저 불러오기
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 입력 데이터 준비
input_text = "Hello, I am analyzing the importance of weights and activations in the llama2 model."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Gradient-based 중요도 계산
model.zero_grad()
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
loss.backward()

gradient_importance = {}
for name, param in tqdm(model.named_parameters(), desc="Gradient-based Importance"):
    if "layers" in name:
        gradient_importance[name] = param.grad.abs().mean().item()

# 민감도 기반 중요도 계산
def sensitivity_importance(model, input_ids, epsilon=1e-5):
    model.zero_grad()
    outputs = model(input_ids, labels=input_ids)
    original_loss = outputs.loss.detach().clone()

    importance_dict = {}
    for name, param in tqdm(model.named_parameters(), desc="Sensitivity-based Importance"):
        if "layers" in name:
            param_clone = param.detach().clone()
            param.data.add_(epsilon)
            perturbed_outputs = model(input_ids, labels=input_ids)
            perturbed_loss = perturbed_outputs.loss
            importance_dict[name] = (perturbed_loss - original_loss).abs().item()
            param.data.copy_(param_clone)

    return importance_dict

sensitivity_importance_dict = sensitivity_importance(model, input_ids)

# 레이어별 중요도 계산
layer_outputs = {}

def get_layer_output(name):
    def hook(model, input, output):
        if isinstance(output, tuple):
            output_list = []
            for o in output:
                if isinstance(o, torch.Tensor):
                    output_list.append(o.detach())
                elif o is None:
                    output_list.append(None)
                else:
                    output_list.append(o)  # DynamicCache 객체 등 다른 타입은 그대로 추가
            layer_outputs[name] = tuple(output_list)
        else:
            layer_outputs[name] = output.detach()
    return hook

for name, layer in model.named_modules():
    if "layers" in name:
        layer.register_forward_hook(get_layer_output(name))

with torch.no_grad():
    outputs = model(input_ids)

layer_importance = {}
for name, output in tqdm(layer_outputs.items(), desc="Layer-wise Importance"):
    if isinstance(output, tuple):
        output_tensor = None
        for o in output:
            if isinstance(o, torch.Tensor):
                output_tensor = o
                break
        if output_tensor is not None:
            min_size = min(output_tensor.view(-1).size(0), outputs.logits.view(-1).size(0))
            correlation, _ = spearmanr(output_tensor.view(-1)[:min_size].cpu().numpy(), outputs.logits.view(-1)[:min_size].cpu().numpy())
            layer_importance[name] = correlation
    elif isinstance(output, torch.Tensor):
        min_size = min(output.view(-1).size(0), outputs.logits.view(-1).size(0))
        correlation, _ = spearmanr(output.view(-1)[:min_size].cpu().numpy(), outputs.logits.view(-1)[:min_size].cpu().numpy())
        layer_importance[name] = correlation

# 그래프 저장
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.bar(gradient_importance.keys(), gradient_importance.values())
plt.title("Gradient-based Importance")
plt.xticks(rotation=90)

plt.subplot(132)
plt.bar(sensitivity_importance_dict.keys(), sensitivity_importance_dict.values())
plt.title("Sensitivity-based Importance")
plt.xticks(rotation=90)

plt.subplot(133)
plt.bar(layer_importance.keys(), layer_importance.values())
plt.title("Layer-wise Importance")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("gpt2_importance_analysis.png")  # 그래프를 PNG 파일로 저장
print("Importance analysis completed. Graph saved as 'gpt2_importance_analysis.png'.")
