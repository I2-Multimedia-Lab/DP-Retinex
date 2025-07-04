import random
import torch

def update_patch_and_batch_sizes(current_iter, groups, mini_gt_sizes, mini_batch_sizes, logger, logger_j, train_data, batch_size, gt_size, scale):
    """
    根据当前迭代次数更新patch大小和batch大小。

    参数:
    - current_iter: 当前的迭代次数
    - groups: 阶段分组
    - mini_gt_sizes: 每个阶段对应的目标图像大小
    - mini_batch_sizes: 每个阶段对应的batch大小
    - logger: 日志记录器，用于输出信息
    - logger_j: 记录阶段更新状态的标志
    - train_data: 包含训练数据的字典，'lq'和'gt'为低质量和真实图像
    - batch_size: 默认的batch大小
    - gt_size: 目标图像的原始大小
    - scale: 超分辨率缩放因子

    返回:
    - 更新后的低质量图像(lq)和真实图像(gt)
    """

    # ------Progressive learning ---------------------
    # 根据当前的iter次数判断在哪个阶段
    j = ((current_iter > groups) != True).nonzero()[0]
    if len(j) == 0:
        bs_j = len(groups) - 1  # 如果当前迭代次数大于所有阶段，使用最后一个阶段
    else:
        bs_j = j[0]  # 获取当前阶段索引

    mini_gt_size = mini_gt_sizes[bs_j]  # 当前阶段的目标图像大小
    mini_batch_size = mini_batch_sizes[bs_j]  # 当前阶段的batch大小

    # 如果logger_j标志为真，则输出更新信息
    if logger_j[bs_j]:
        logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(
            mini_gt_size, mini_batch_size * torch.cuda.device_count()))
        logger_j[bs_j] = False  # 更新标志为假，避免重复日志

    lq = train_data['lq']  # 获取低质量图像
    gt = train_data['gt']  # 获取真实图像

    # 处理batch_size不足的情况
    if mini_batch_size < batch_size:
        indices = random.sample(range(0, batch_size), k=mini_batch_size)  # 随机抽样
        lq = lq[indices]  # 低质量图像抽样
        gt = gt[indices]  # 真实图像抽样

    # 超分辨任务下的操作，更新图像尺寸
    if mini_gt_size < gt_size:
        x0 = int((gt_size - mini_gt_size) * random.random())  # 随机选择切片起始点
        y0 = int((gt_size - mini_gt_size) * random.random())
        x1 = x0 + mini_gt_size  # 结束点
        y1 = y0 + mini_gt_size

        # 根据新的坐标切片低质量和真实图像
        lq = lq[:, :, x0:x1, y0:y1]
        gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]

    # -------------------------------------------
    return lq, gt  # 返回更新后的低质量和真实图像
