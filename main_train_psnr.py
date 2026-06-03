# ============================================================================
# 导入依赖
# ============================================================================
import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset    # 数据集工厂：根据配置的 dataset_type 创建对应 Dataset
from models.select_model import define_Model       # 模型工厂：根据配置的 model 类型创建对应 Model


'''
# --------------------------------------------
# PSNR 导向训练脚本（像素级损失，不含GAN）
# 支持模型：MSRResNet, DnCNN, FFDNet, SRMD, DPSR, RRDB, IMDN, SwinIR, DRUNet 等
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    # ========================================================================
    # Step 1：准备配置（prepare opt）
    #   - 解析命令行参数和 JSON 配置文件
    #   - 自动查找最新 checkpoint 实现断点续训
    #   - 配置日志、随机种子
    # ========================================================================

    # 1.1 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')  # 'pytorch' 或 'slurm'
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)  # 是否启用分布式训练

    # 1.2 解析 JSON 配置文件（去掉 // 注释、填默认值、展开路径、设置 CUDA_VISIBLE_DEVICES）
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # 1.3 初始化分布式训练（如果启用）
    if opt['dist']:
        init_dist('pytorch')                     # 初始化 NCCL 进程组
    opt['rank'], opt['world_size'] = get_dist_info()  # 获取当前进程 rank 和总进程数

    # 1.4 创建输出目录（仅 rank 0 进程执行，避免多进程竞争）: {root}/{task}/models/, images/, options/
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # 1.5 自动查找最新 checkpoint，实现断点续训
    #     扫描 {root}/{task}/models/ 目录，按文件名中的 iteration 号找最新的 .pth 文件
    #     文件名格式: {iteration}_G.pth, {iteration}_E.pth, {iteration}_optimizerG.pth
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G       # 生成器权重路径
    opt['path']['pretrained_netE'] = init_path_E       # EMA 模型权重路径
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG  # 优化器状态路径
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)  # 取最大值作为当前步数

    border = opt['scale']  # PSNR 计算时裁剪边缘的像素数（避免边界效应影响评估）

    # 1.6 保存本次运行的配置快照到 {root}/{task}/options/ 目录（仅 rank 0）
    if opt['rank'] == 0:
        option.save(opt)

    # 1.7 将 OrderedDict 转为 NoneDict：访问不存在的 key 返回 None 而非抛 KeyError
    opt = option.dict_to_nonedict(opt)

    # 1.8 配置 logger（仅 rank 0 写日志）
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))  # 打印完整配置

    # 1.9 设置随机种子，保证可复现性
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)     # 未指定则随机生成
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ========================================================================
    # Step 2：创建 DataLoader（create dataloader）
    #   - 根据 dataset_type 通过工厂方法创建对应的 Dataset
    #   - 分别构建训练集和测试集的 DataLoader
    # ========================================================================

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # 2.1 训练集：根据 dataset_type（如 "sr"、"dncnn"）工厂创建 Dataset
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                # 分布式模式：用 DistributedSampler 确保每个 GPU 拿不同数据
                # batch_size 和 num_workers 都会被 num_gpu 均分
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,           # 用了 sampler 就不能 shuffle
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,          # 丢弃不完整的最后一个 batch
                                          pin_memory=True,         # 加速 CPU→GPU 数据传输
                                          sampler=train_sampler)
            else:
                # 单机模式：DataParallel 或单 GPU
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            # 2.2 测试集：batch_size=1，按张图推理并计算 PSNR
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # ========================================================================
    # Step 3：初始化模型（initialize model）
    #   - 工厂创建 Model（ModelPlain/ModelPlain2/ModelPlain4/ModelGAN/ModelVRT）
    #   - init_train() 内部：加载预训练权重 → 定义损失函数 → 定义优化器 → 定义学习率调度器
    # ========================================================================

    model = define_Model(opt)
    model.init_train()  # 执行 load() → define_loss() → define_optimizer() → define_scheduler()
    if opt['rank'] == 0:
        logger.info(model.info_network())  # 打印网络结构和参数量
        logger.info(model.info_params())   # 打印每层参数统计（均值/最大/最小/标准差）

    # ========================================================================
    # Step 4：训练主循环（main training loop）
    #   - 外层无限 epoch 循环，内层遍历 DataLoader
    #   - 每个 iteration：更新学习率 → 前向传播 → 计算损失 → 反向传播 → 梯度下降
    #   - 定期：打印日志、保存 checkpoint、在验证集上测试
    # ========================================================================

    for epoch in range(1000000):  # 无限 epoch 循环（没有退出条件，手动停止）
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)  # 分布式下每个 epoch 用不同 shuffle 顺序

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # ------------------------------------------------------------------
            # 4.1 更新学习率：调度器按 MultiStepLR 或 CosineAnnealing 策略衰减
            # ------------------------------------------------------------------
            model.update_learning_rate(current_step)

            # ------------------------------------------------------------------
            # 4.2 喂数据：将 L（低质量图）和 H（高质量真值）从 CPU 搬到 GPU
            # ------------------------------------------------------------------
            model.feed_data(train_data)

            # ------------------------------------------------------------------
            # 4.3 参数优化（核心步骤）：
            #     zero_grad → netG(L) 得到 E → 计算 loss(E, H) → backward → clip_grad → optimizer.step
            #     可选：正交正则化、权重裁剪、EMA 更新
            # ------------------------------------------------------------------
            model.optimize_parameters(current_step)

            # ------------------------------------------------------------------
            # 4.4 打印训练信息（每 checkpoint_print 步，仅 rank 0）
            # ------------------------------------------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # 获取 loss 等日志 dict
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # 拼接所有日志信息
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # ------------------------------------------------------------------
            # 4.5 保存模型 checkpoint（每 checkpoint_save 步，仅 rank 0）
            #     保存文件：{iter}_G.pth, {iter}_E.pth, {iter}_optimizerG.pth
            # ------------------------------------------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # ------------------------------------------------------------------
            # 4.6 验证集测试（每 checkpoint_test 步，仅 rank 0）
            #     遍历测试集每张图 → 推理 → 保存结果 → 计算 PSNR → 输出平均 PSNR
            # ------------------------------------------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    # 为每张测试图创建独立子目录，存放不同 iteration 的结果
                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    # 推理：feed_data → netG_forward（开启 eval 模式 + no_grad）
                    model.feed_data(test_data)
                    model.test()

                    # 获取估计图 E 和真值 H（tensor → uint8 numpy）
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # 保存估计图：{img_name}_{current_step}.png
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # 计算 PSNR（裁剪 border 像素避免边界效应）
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # 输出本轮测试的平均 PSNR
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
