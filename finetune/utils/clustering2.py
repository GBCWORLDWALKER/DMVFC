import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
def clustering_function(x, weights=None, alpha=1.0, in_features=10, out_features=10, FA_base_path=None, fiber_array=None):
    """
    执行聚类层计算，并根据个体ID顺序加载对应的FA文件
    
    参数:
        x: 输入张量，形状为 [batch_size, in_features]，这里batch_size为40000(20个体*2000根纤维)
        weights: 聚类中心权重，形状为 [out_features, in_features]
        alpha: t-分布参数，默认为1.0
        in_features: 输入特征维度
        out_features: 输出特征维度（聚类数量）
        FA_base_path: FA文件路径的基础部分，例如'/data/FA'
        fiber_array: 包含纤维信息的对象
    """
    
    # 确定每个纤维所属的个体ID
    unique_subjects = np.unique(fiber_array.fiber_subID)
    fibers_per_subject = 2000  # 每个个体的纤维数量
    
    # 加载所有个体的FA图像
    FA_images = {}
    FA_volumes = {}
    
    for subject_id in unique_subjects:
        # 构建FA文件路径
        FA_path = f"{FA_base_path}/sub_{subject_id:02d}/dti__FA.nii.gz"
        
        # 检查文件是否存在
        if not os.path.exists(FA_path):
            raise FileNotFoundError(f"找不到FA文件: {FA_path}")
        
        # 加载FA图像
        FA_img = nib.load(FA_path)
        FA_images[subject_id] = FA_img
        FA_volumes[subject_id] = FA_img.get_fdata()
        print(f"已加载FA图像: {FA_path}")
    
    # FMRI与FA图像空间转换的仿射矩阵
    FMRI_affine = np.array([
        [-2.0,  0.0,  0.0,  90.0],
        [ 0.0,  2.0,  0.0, -126.0],
        [ 0.0,  0.0,  2.0, -72.0],
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    
    # 为每个纤维计算FA特征
    fiber_FA_values = []
    for i, fiber_id in enumerate(range(fiber_array.fiber_array_ras.shape[0])):
        # 获取该纤维的个体ID
        subject_id = fiber_array.fiber_subID[fiber_id]
        
        # 获取该个体的FA图像和仿射矩阵
        FA_img = FA_images[subject_id]
        FA_affine = FA_img.affine
        FA_volume = FA_volumes[subject_id]
        
        # 获取该纤维的所有点坐标
        fiber_points = fiber_array.fiber_array_ras[fiber_id]
        
        # 将纤维坐标从RAS空间转换到体素空间
        p_voxel = nib.affines.apply_affine(np.linalg.inv(FA_affine), 
                                           nib.affines.apply_affine(FMRI_affine, fiber_points))
        
        # 对每个点提取FA值
        FA_values = []
        for point in p_voxel:
            FA_value = extract_FA(point, FA_volume)
            FA_values.append(FA_value)
        
        # 存储该纤维的FA值
        fiber_FA_values.append(FA_values)
    
    # 原始聚类计算
    if weights is None:
        weights = torch.nn.init.xavier_uniform_(torch.Tensor(out_features, in_features))
    
    # 标准化处理
    x_normalized = torch.mul(F.normalize(x, p=2, dim=1), 100)
    weight_normalized = torch.mul(F.normalize(weights, p=2, dim=1), 100)
    
    # 计算欧氏距离
    x_diff = x_normalized.unsqueeze(1) - weight_normalized
    x_squares = torch.mul(x_diff, x_diff)
    x_dist = torch.sum(x_squares, dim=2)
    orig_dist = x_dist.clone()  # 保存原始距离矩阵
    
    # 将FA值转换为张量
    fiber_FA_tensor = torch.tensor(fiber_FA_values, device=x.device).float()
    
    # 为每个聚类中心找到最接近的纤维
    closest_fibers = []
    for c in range(weight_normalized.shape[0]):  # 遍历每个聚类中心
        distances = orig_dist[:, c]
        closest_idx = torch.argmin(distances).item()
        closest_fibers.append(closest_idx)
    
    # 计算每个纤维与每个聚类中心代表纤维的FA距离
    fa_distance_matrix = torch.zeros_like(orig_dist)

    for c, rep_idx in enumerate(closest_fibers):
        rep_fa = fiber_FA_tensor[rep_idx]  # 代表纤维的FA值
        
        # 直接计算FA距离而不是相似度
        # 计算每个点的FA绝对差异并取平均值作为距离
        fa_diff = torch.abs(fiber_FA_tensor - rep_fa.unsqueeze(0))
        fa_distance = fa_diff.mean(dim=1)  # 沿点维度取平均，得到每个纤维到代表纤维的FA距离
        
        # 存储到距离矩阵中
        fa_distance_matrix[:, c] = fa_distance

    # 归一化FA距离到与几何距离相同的范围
    # 计算原始距离的统计量
    orig_min = orig_dist.min()
    orig_max = orig_dist.max()
    orig_mean = orig_dist.mean()
    orig_std = orig_dist.std()

    # 计算FA距离的统计量
    fa_min = fa_distance_matrix.min()
    fa_max = fa_distance_matrix.max()
    fa_mean = fa_distance_matrix.mean()
    fa_std = fa_distance_matrix.std()

    print(f"空间距离平均值: {orig_mean.item():.4f}")
    print(f"FA距离平均值: {fa_mean.item():.4f}")

    beta = orig_mean / fa_mean

    beta1 = beta / 200
    
    # # 1. 标准化FA距离
    # standardized_fa = (fa_distance_matrix - fa_mean) / (fa_std + 1e-8)
    
    # # 2. 使用温度系数调整softmax的平滑度
    # temperature = 0.5  # 可调整参数
    # softmax_input = -standardized_fa / temperature
    
    # # 3. 对每个纤维的所有聚类距离应用softmax
    # softmax_fa = F.softmax(softmax_input, dim=1)

    # 直接将几何距离和FA距离加权相加
    fa_weight = beta1  # 控制FA距离的权重，可以调整
    combined_dist = orig_dist + (fa_distance_matrix *  35)
    
    # scaled_fa_dist = softmax_fa + 1e-8
    
    # # 可选：添加一个指数参数来控制FA距离的影响
    # fa_exp = 0.1  # 小于1的值会减弱FA距离的影响，大于1的值会增强FA距离的影响
    # combined_dist = orig_dist * (scaled_fa_dist ** fa_exp)

    # 使用组合后的距离继续t-SNE计算
    x_dis = combined_dist  # 保存组合后的距离矩阵用于返回
    x = 1.0 + (combined_dist / alpha)
    x = 1.0 / x
    x = x ** ((alpha + 1.0) / 2.0)
    x = torch.t(x) / torch.sum(x, dim=1)
    x = torch.t(x)
    
    return x, x_dis, fiber_FA_tensor  # 返回聚类结果、距离和FA值

def extract_FA(point, FA_volume):
    """使用三线性插值从FA体积中提取给定坐标点的FA值
    
    参数:
    point: 包含x,y,z坐标的数组或列表
    FA_volume: 3D体积FA数据
    
    返回:
    插值后的FA值
    """
    p_x, p_y, p_z = point
    
    # 确保坐标在有效范围内
    p_x = max(0, min(p_x, FA_volume.shape[0]-1.001))
    p_y = max(0, min(p_y, FA_volume.shape[1]-1.001))
    p_z = max(0, min(p_z, FA_volume.shape[2]-1.001))
    
    # 获取坐标的整数部分和小数部分
    x0, y0, z0 = int(np.floor(p_x)), int(np.floor(p_y)), int(np.floor(p_z))
    x1, y1, z1 = min(x0 + 1, FA_volume.shape[0]-1), min(y0 + 1, FA_volume.shape[1]-1), min(z0 + 1, FA_volume.shape[2]-1)
    
    # 计算插值权重
    wx = p_x - x0
    wy = p_y - y0
    wz = p_z - z0
    
    # 提取8个顶点的FA值
    v000 = FA_volume[x0, y0, z0]
    v001 = FA_volume[x0, y0, z1]
    v010 = FA_volume[x0, y1, z0]
    v011 = FA_volume[x0, y1, z1]
    v100 = FA_volume[x1, y0, z0]
    v101 = FA_volume[x1, y0, z1]
    v110 = FA_volume[x1, y1, z0]
    v111 = FA_volume[x1, y1, z1]
    
    # 先在x方向插值
    c00 = v000 * (1 - wx) + v100 * wx
    c01 = v001 * (1 - wx) + v101 * wx
    c10 = v010 * (1 - wx) + v110 * wx
    c11 = v011 * (1 - wx) + v111 * wx
    
    # 再在y方向插值
    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy
    
    # 最后在z方向插值
    c = c0 * (1 - wz) + c1 * wz
    
    return c