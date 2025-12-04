import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
from torch.utils.data import SubsetRandomSampler

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info

# Autonomous Driving Scenario Reconstruction Dataset Compression Using Loss Sort

def train(version,
          dataroot='/data/nuscenes',
          nepochs=10000,
          gpuid=1,

          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-50.0, 50.0, 0.5],
          ybound=[-50.0, 50.0, 0.5],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          bsz=4,
          nworkers=10,
          lr=1e-3,
          weight_decay=1e-7,
          ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    # Calculate the total number of training samples
    total_train_samples = len(trainloader.dataset)

    # Generate a list of indices
    indices = list(range(total_train_samples))
    # Shuffle the indices randomly
    np.random.shuffle(indices)

    # Create a SubsetRandomSampler for the selected training indices
    train_sampler = SubsetRandomSampler(indices[:int(0.9 * total_train_samples)])

    # Create a new DataLoader with the subset of data (90%)
    trainloader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=bsz, sampler=train_sampler,
                                              num_workers=nworkers, drop_last=True)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        sample_losses = []  # To store loss values for each sample
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                          rots.to(device),
                          trans.to(device),
                          intrins.to(device),
                          post_rots.to(device),
                          post_trans.to(device),
                          )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            # Track the loss for each sample
            sample_losses.extend(loss.item() for _ in range(len(imgs)))

            counter += 1
            t1 = time()

            # Open the file in append mode (so it doesn't overwrite previous content)
            with open("training_log.txt", "a") as log_file:
                if counter % 10 == 0:
                    # Print the counter and loss to the console
                    print(counter, loss.item())

                    # Write the same information to the log file
                    log_file.write(f"{counter} {loss.item()}\n")

                    # Log the scalar loss value to TensorBoard
                    writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

        # After each epoch, select the 90% of samples with the lowest loss values
        sample_loss_indices = np.argsort(sample_losses)[:int(0.9 * total_train_samples)]
        train_sampler = SubsetRandomSampler(sample_loss_indices)
        trainloader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=bsz, sampler=train_sampler,
                                                  num_workers=nworkers, drop_last=True)

        # After each epoch, select the 85% of samples with the highest loss values
        # sample_loss_indices = np.argsort(sample_losses)[-int(0.85 * total_train_samples):]  # 选择最大损失值的85%
        # train_sampler = SubsetRandomSampler(sample_loss_indices)
        # trainloader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=bsz, sampler=train_sampler,num_workers=nworkers, drop_last=True)
