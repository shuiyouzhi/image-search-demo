# export_clip_model_fixed.py
import torch
import clip
import onnx

# 1. 加载CLIP模型
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 2. 定义视觉编码器
class CLIPVisionEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        
    def forward(self, x):
        features = self.visual(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

vision_encoder = CLIPVisionEncoder(model)
vision_encoder.eval()

# 3. 测试输出维度
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = vision_encoder(dummy_input)
    print(f"输出维度: {output.shape}")

# 4. 导出ONNX - 使用opset_version=14（支持更多操作符）
print("正在导出模型...")

torch.onnx.export(
    vision_encoder,
    dummy_input,
    "clip_vision_feature.onnx",
    input_names=['input'],
    output_names=['feature'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'feature': {0: 'batch_size'}
    },
    opset_version=14,  # 改为14，支持unflatten等操作符
    do_constant_folding=True,
    export_params=True,
    external_data=False  # 导出为单文件
)

print("✅ CLIP模型导出成功！")

# 5. 验证并清理
import os
if os.path.exists("clip_vision_feature.onnx.data"):
    os.remove("clip_vision_feature.onnx.data")
    print("已删除外部数据文件")

# 6. 验证模型
try:
    onnx_model = onnx.load("clip_vision_feature.onnx")
    onnx.checker.check_model(onnx_model)
    file_size = os.path.getsize("clip_vision_feature.onnx") / (1024 * 1024)
    print(f"✅ 验证通过！文件大小: {file_size:.2f} MB")
except Exception as e:
    print(f"⚠️ 验证失败: {e}")