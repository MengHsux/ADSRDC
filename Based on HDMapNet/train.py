import os
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse
from torch.utils.data import SubsetRandomSampler

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from loss import SimpleLoss, DiscriminativeLoss

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from evaluate import onehot_encoding, eval_iou
from model.pointpillar import PointPillar


# Autonomous Driving Scenario Reconstruction Dataset Compression Using Loss Sort

def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


def train(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    # 获取原始训练和验证数据加载器
    # 注意：这里我们假设semantic_dataset返回的是(train_loader, val_loader)
    # 而不是(train_dataset, val_loader)
    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)

    # 检查train_loader是否有dataset属性
    if hasattr(train_loader, 'dataset'):
        train_dataset = train_loader.dataset
        total_train_samples = len(train_dataset)
        logger.info(f"Total training samples: {total_train_samples}")

        # 生成索引列表并随机打乱
        indices = list(range(total_train_samples))
        np.random.shuffle(indices)

        # 初始使用90%的数据进行训练
        train_sampler = SubsetRandomSampler(indices[:int(0.9 * total_train_samples)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, sampler=train_sampler,
                                                   num_workers=args.nworkers, drop_last=True)
    else:
        # 如果train_loader没有dataset属性，说明无法数据集剪枝功能
        logger.warning("Warning: train_loader does not have 'dataset' attribute. Dataset pruning disabled.")
        # 使用原始的train_loader，不进行数据集剪枝
        total_train_samples = 0

    if args.model == 'HDMapNet_lidar':
        model = PointPillar(NUM_CLASSES + 1, args.xbound, args.ybound, args.zbound, args.embedding_dim)
    else:
        model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred,
                          args.angle_class)
    if args.finetune:
        model.load_state_dict(torch.load(args.modelf), strict=False)
        for name, param in model.named_parameters():
            if 'bevencode.up' in name or 'bevencode.layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 10, 0.1)
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')

    model.train()
    counter = 0
    last_idx = len(train_loader) - 1 if hasattr(train_loader, '__len__') else 0

    for epoch in range(args.nepochs):
        # 存储每个样本的损失值
        sample_losses = []

        # 如果可以进行数据集剪枝，获取当前采样器的索引
        current_indices = []
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, SubsetRandomSampler):
            current_indices = train_loader.sampler.indices

        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()

            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                   post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                   lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())

            semantic_gt = semantic_gt.cuda().float()
            instance_gt = instance_gt.cuda()

            # 计算每个样本的分割损失
            batch_seg_loss = F.binary_cross_entropy_with_logits(semantic, semantic_gt, reduction='none')
            batch_seg_loss = batch_seg_loss.mean(dim=[1, 2, 3])  # 对空间维度求平均

            # 计算批次的分割损失
            seg_loss = batch_seg_loss.mean()

            if args.instance_seg:
                # 计算实例分割损失
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
                # 假设这些损失已经是批次平均
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            # 计算每个样本的总损失
            batch_total_loss = batch_seg_loss * args.scale_seg

            # 如果有实例分割损失，将其平均分配到每个样本
            if args.instance_seg:
                batch_total_loss += (var_loss * args.scale_var + dist_loss * args.scale_dist) / len(imgs)

            # 如果有方向预测损失，将其平均分配到每个样本
            if args.direction_pred:
                batch_total_loss += direction_loss * args.scale_direction / len(imgs)

            # 计算批次的平均损失
            final_loss = batch_total_loss.mean()

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            # 保存当前批次每个样本的损失
            sample_losses.extend(batch_total_loss.detach().cpu().numpy())

            if counter % 10 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1 - t0:>7.4f}    "
                            f"Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)

        # 检查是否可以进行数据集剪枝
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, SubsetRandomSampler) and \
                hasattr(train_loader, 'dataset') and len(sample_losses) > 0:

            # 确保损失值和索引数量匹配
            if len(sample_losses) == len(current_indices):
                # 按损失值排序并选择损失最低的90%
                sorted_indices = np.argsort(sample_losses)[:int(0.9 * len(sample_losses))]
                selected_indices = [current_indices[i] for i in sorted_indices]

                # 创建新的采样器
                train_sampler = SubsetRandomSampler(selected_indices)
                # 更新训练数据加载器
                train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.bsz,
                                                           sampler=train_sampler, num_workers=args.nworkers,
                                                           drop_last=True)
                last_idx = len(train_loader) - 1

                logger.info(f"EPOCH[{epoch:>3d}]: Selected {len(selected_indices)} samples (90% with lowest loss)")
            else:
                logger.warning(
                    f"Sample losses count ({len(sample_losses)}) doesn't match indices count ({len(current_indices)}). Skipping pruning.")
        elif hasattr(train_loader, 'dataset'):
            # 如果有dataset但没有sampler，创建一个新的sampler
            if len(sample_losses) > 0 and len(sample_losses) == len(train_loader.dataset):
                # 按损失值排序并选择损失最低的90%
                sorted_indices = np.argsort(sample_losses)[:int(0.9 * len(sample_losses))]

                # 创建新的采样器
                train_sampler = SubsetRandomSampler(sorted_indices)
                # 更新训练数据加载器
                train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.bsz,
                                                           sampler=train_sampler, num_workers=args.nworkers,
                                                           drop_last=True)
                last_idx = len(train_loader) - 1

                logger.info(f"EPOCH[{epoch:>3d}]: Selected {len(sorted_indices)} samples (90% with lowest loss)")
            else:
                logger.warning("Dataset pruning not possible in this epoch.")
        else:
            logger.info("Dataset pruning disabled.")

        iou = eval_iou(model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

        write_log(writer, iou, 'eval', counter)

        # Save model only at the last epoch
        if epoch == args.nepochs - 1:
            model_name = os.path.join(args.logdir, f"model.pt")
            torch.save(model.state_dict(), model_name)
            logger.info(f"{model_name} saved")

        model.train()
        sched.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training with dataset pruning.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_fusion')

    # training config
    parser.add_argument("--nepochs", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', default=True)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', default=True)
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()

    # 添加错误捕获和日志记录
    try:
        train(args)
    except Exception as e:
        # 创建错误日志文件
        error_log_path = os.path.join(args.logdir, "error_log.txt")
        with open(error_log_path, 'w') as f:
            f.write(f"Error occurred at {time()}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            f.write(f"Error message: {str(e)}\n")
            f.write("\nTraceback:\n")
            import traceback

            traceback.print_exc(file=f)

        print(f"An error occurred. Please check the error log at: {error_log_path}")
        raise e  # 重新抛出异常常，以便用户在控制台也能看到
