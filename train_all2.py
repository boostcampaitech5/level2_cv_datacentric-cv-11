import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True     #활성화시 속도 감소 이슈 있으므로 배포시 활성화
    # torch.backends.cudnn.benchmark = False    
    np.random.seed(seed)      #dataset 초기화 시 np.random사용
    random.seed(seed) 
    
def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--ufo_name', default='train') #추가
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--max_epoch', type=int, default=50)

    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    return args

def wandb_config(args):
    config_dict  = {'data_dir'      : args.data_dir,
                    'ufo_name'      : args.ufo_name,
                    'image_size'    : args.image_size,
                    'input_size'    : args.input_size,
                    'batch_size'    : args.batch_size,
                    'learning_rate' : args.learning_rate,
                    'max_epoch'     : args.max_epoch}
    return config_dict

def do_training(data_dir, model_dir, device, ufo_name, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, ignore_tags, seed):
    seed_everything(args.seed)
    dataset = SceneTextDataset(
        data_dir,
        ufo_name =ufo_name,
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        rotate = True,
        brightness_contrast = False,
        clahe = False,
        motion_blur = True,
        all_aug = True
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    wandb.init(config=wandb_config(args),project='Data-Centric', entity='cv11_aivengers',name=f'{args.ufo_name}_epoch={args.max_epoch}_all2_512') #수정
    
    model.train()
    val_loss = float("inf")
    # model_dir 수정
    wandb_name = f'{args.ufo_name}_epoch={args.max_epoch}_all2_512'
    model_dir = osp.join(model_dir, wandb_name)
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                total_loss={'total_loss': sum(val_dict.values())}
                wandb.log(val_dict, step = epoch)
                wandb.log(total_loss,step=epoch)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        mean_loss={'mean_loss':epoch_loss / num_batches}
        wandb.log(mean_loss,step=epoch)

        now_val_loss = epoch_loss / num_batches

        # 매 에폭마다 latest 저장
        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        ckpt_fpath_latest = osp.join(model_dir, 'latest.pth')
        torch.save(model.state_dict(), ckpt_fpath_latest)

        if val_loss > now_val_loss:
            val_loss = now_val_loss
            file_list = os.listdir(model_dir)
            # best가 있는 pth 파일만 따로 리스트
            best_file_path = [i for i in file_list if 'best' in i]
            if best_file_path: # 만약에 best가 있으면
                # 기존의 best를 삭제
                os.remove(osp.join(model_dir, best_file_path[0]))
            ckpt_fpath = osp.join(model_dir, f'best_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
def main(args):
    print(args.__dict__)
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
