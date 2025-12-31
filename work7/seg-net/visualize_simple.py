"""
简单可视化工具：加载模型并对指定数据进行分割可视化
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch

from lib.models.seg_net_lite import get_seg_net
from lib.models.fc_autoencoder import get_fc_autoencoder
from lib.utils.vis import vis_segments


def main():
    parser = argparse.ArgumentParser(description='简单分割可视化工具')
    parser.add_argument('--model', type=str, default='segnet',
                        choices=['segnet', 'fc'],
                        help='模型类型: segnet 或 fc')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--data', type=str, required=True,
                        help='数据文件路径 (.mat文件)')
    parser.add_argument('--save', type=str, default=None,
                        help='保存图片路径（可选）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("简单分割可视化工具")
    print("="*60)
    
    # 1. 加载模型
    print(f"\n加载模型: {args.model}")
    if args.model == 'segnet':
        model = get_seg_net()
    elif args.model == 'fc':
        model = get_fc_autoencoder()
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"  ✓ 模型加载成功: {args.checkpoint}")
    
    # 2. 加载数据
    print(f"\n加载数据: {args.data}")
    data = loadmat(args.data)
    image = data['imgMat'].transpose([3, 2, 0, 1]).astype(np.float32)
    semantic_mask = data['semanticMaskMat'].transpose([2, 0, 1]).astype(np.int64)
    
    # 归一化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    image_norm = (image - mean) / std
    print(f"  ✓ 数据加载成功，共 {image.shape[0]} 张图像")
    
    # 3. 预测
    print(f"\n开始预测...")
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_norm)
        output = model(image_tensor)
        predictions = output.argmax(dim=1).numpy()
    print(f"  ✓ 预测完成")
    
    # 4. 可视化第一张图像
    print(f"\n可视化结果...")
    img_display = image[0].transpose(1, 2, 0)
    gt_mask = semantic_mask[0]
    pred_mask = predictions[0]
    
    # 计算mIoU
    miou = calculate_miou(gt_mask, pred_mask, 11)
    
    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(vis_segments(gt_mask, 11))
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(vis_segments(pred_mask, 11))
    axes[2].set_title(f'Prediction (mIoU: {miou:.3f})', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'Segmentation Result - {args.model.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"  ✓ 结果已保存: {args.save}")
    else:
        plt.show()
    
    print(f"\n性能: mIoU = {miou:.4f}")
    print("="*60)


def calculate_miou(gt, pred, n_class):
    """计算mIoU"""
    mask = (gt >= 0) & (gt < n_class)
    hist = np.bincount(
        n_class * gt[mask].astype(int) + pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0
    mean_iu = np.nanmean(iu[valid])
    
    return mean_iu

# python visualize_simple.py     \
#         --model segnet     \
#         --checkpoint out/checkpoint.pth.tar     \
#         --data data/multi-digit-mnist/testset001.mat     \
#         --save segnet_result.png
if __name__ == '__main__':
    main()
