
from torch.utils.data import DataLoader
import os
import builtins
import argparse
import torch
import torch.distributed as dist
import time
import FSPNet_model
import dataset
import loss
from torch.amp import GradScaler, autocast


def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--batch_size_per_gpu', default=3, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--img_size', default=384, type=int, help='input image size (square)')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader workers')
    parser.add_argument('--prefetch_factor', default=2, type=int, help='dataloader prefetch factor (workers > 0)')
    parser.add_argument('--persistent_workers', action='store_true', help='keep dataloader workers alive (workers > 0)')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='enable cudnn benchmark for fixed-size inputs')
    parser.add_argument('--output_dir', default='./checkpoints', type=str, help='where to save checkpoints')
    parser.add_argument('--save_every', default=2, type=int, help='save every N epochs')
    parser.add_argument('--save_after', default=30, type=int, help='start saving after this epoch')
    parser.add_argument('--amp', action='store_true', help='use mixed precision training')
    parser.add_argument('--accum_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--path', type=str, help='path to train dataset')
    parser.add_argument('--pretrain', type=str, help='path to pretrain model')
    parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')
    
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args
                                         
def main(args):
    # DDP setting (supports torchrun/torch.distributed.run)
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    if args.rank == -1 and "RANK" in os.environ:
        args.rank = int(os.environ["RANK"])
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    if args.distributed:
        # Prefer torchrun env/configs
        if args.local_rank != -1:
            args.gpu = args.local_rank
            if args.rank == -1:
                args.rank = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            print("args.rank = {}; args.gpu = {}".format(args.rank, args.gpu))
        else:
            # Fallback: single-process distributed misconfig protection
            args.gpu = 0
            if args.rank == -1:
                args.rank = 0
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Single-GPU fallback
    if not args.distributed:
        args.rank = 0
        if args.gpu is None:
            args.gpu = 0

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # suppress printing if not on master gpu
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    net = FSPNet_model.Model(args.pretrain, img_size=args.img_size)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    net = net.to(device)

    if args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        if args.rank == 0:
            os.system("nvidia-smi")
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
        model_without_ddp = net.module
    else:
        model_without_ddp = net
        
    ### optimizer ###
    

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    encoder_param=[]
    decoer_param=[]
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    # optimizer = torch.optim.SGD([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}], momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}])
    
    ### resume training if necessary ###
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    ### Fine tuning for MoCA ###
    if args.ft_for_MoCA is not None:
        ckpt = torch.load(args.ft_for_MoCA, map_location='cpu')
        net.load_state_dict(ckpt)
        print("Fine tuning for MoCA, ckpt from: {}".format(args.ft_for_MoCA))

    
    ### data ###
    Dir = [args.path]
    Dataset = dataset.TrainDataset(Dir)
    if args.distributed:
        Datasampler = torch.utils.data.distributed.DistributedSampler(Dataset, shuffle=True)
        shuffle = False
    else:
        Datasampler = None
        shuffle = True
    collate_fn = lambda batch: dataset.my_collate_fn(batch, size=args.img_size)
    loader_kwargs = dict(
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=Datasampler,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    Dataloader = DataLoader(Dataset, **loader_kwargs)
    
    # torch.backends.cudnn.benchmark = True
    
    ### main loop ###
    star_time=time.time()
    scaler = GradScaler('cuda', enabled=args.amp)

    for curr_epoch in range(args.epochs):
        
        if curr_epoch==50 or curr_epoch==75:
            for param_group in optimizer.param_groups:
                param_group['lr']= param_group['lr']*0.1
                print("Learning rate:", param_group['lr'])
        if Datasampler is not None:
            Datasampler.set_epoch(curr_epoch)
        net.train()
        running_loss_all, running_loss_m , running_loss_edge= 0., 0., 0.
        count = 0
        for data in Dataloader:
            count += 1
            img = data['img'].to(device, non_blocking=True)
            label = data['label'].to(device, non_blocking=True)
            edge = data['edge'].to(device, non_blocking=True)

            with autocast('cuda', enabled=args.amp):
                mask_out, edge_out = net(img)
                all_loss = loss.structure_loss(mask_out, label)
                edge_loss = loss.multi_edge_loss(edge_out, edge)
                total_loss = (all_loss + edge_loss) / max(args.accum_steps, 1)

            if count % max(args.accum_steps, 1) == 1:
                optimizer.zero_grad(set_to_none=True)

            scaler.scale(total_loss).backward()

            if count % max(args.accum_steps, 1) == 0:
                scaler.step(optimizer)
                scaler.update()

        # Handle remainder steps when dataset size isn't divisible by accum_steps
        if count % max(args.accum_steps, 1) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss_all += all_loss.item()
            running_loss_edge += edge_loss.item()

            if count % 20 == 0 and args.rank == 0:
                print("Epoch:{}, Iter:{}, all_loss:{:.5f}, edge_loss:{:.5f}".format(
                    curr_epoch, count, running_loss_all / count, running_loss_edge / count))

        if args.rank == 0 and curr_epoch >= args.save_after and (curr_epoch % args.save_every == 0):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_save_path = os.path.join(args.output_dir, f"model_{curr_epoch}.pth")
            torch.save(model_without_ddp.state_dict(), ckpt_save_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)