# Triton 模型仓库配置
# 目录结构：
#   triton_model_repository/
#   ├── alphacast_resnet/
#   │   ├── config.pbtxt
#   │   └── 1/
#   │       └── model.pt (TorchScript, 由 train_alphacast.py 生成)
#   ├── resnet_encoder/
#   │   ├── config.pbtxt
#   │   └── 1/
#   │       └── model.pt
#   └── lgbm_proxy/
#       ├── config.pbtxt
#       └── 1/
#           └── model.onnx
#
# 启动 Triton:
#   docker-compose up -d triton
#
# 测试健康检查:
#   curl http://localhost:8000/v2/health/ready
#   curl http://localhost:8000/v2/models/alphacast_resnet
