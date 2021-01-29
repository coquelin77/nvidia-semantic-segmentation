"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import time
import torch

from runx.logx import logx
from config import assert_and_infer_cfg, update_epoch, cfg
from utils.misc import AverageMeter, prep_experiment, eval_metrics
from utils.misc import ImageDumper
from utils.trnval_utils import eval_minibatch, validate_topn
from loss.utils import get_loss
from loss.optimizer import get_optimizer, restore_opt, restore_net

import datasets
import network
import torch.cuda.amp as amp
import horovod.torch as hvd
import heat as ht
import pickle

from mpi4py import MPI

# # Import autoresume module
# sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
# AutoResume = None
# try:
#     from userlib.auto_resume import AutoResume
# except ImportError:
#     print(AutoResume)


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
# default network is deepv3.DeepV3PlusW38 or deepv3.DeepV3PlusW38I
parser.add_argument('--arch', type=str, default='deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--dataset_inst', default=None,
                    help='placeholder for dataset instance')
parser.add_argument('--num_workers', type=int, default=6,
                    help='cpu worker threads per dataloader instance')

parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                          ' to 3 in config'))

parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='Use coarse annotations for specific classes')

parser.add_argument('--custom_coarse_dropout_classes', type=str, default=None,
                    help='Drop some classes from auto-labelling')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--rmi_loss', action='store_true', default=False,
                    help='use RMI loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help=('Batch weighting for class (use nll class weighting using '
                          'batch stats'))

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new lr ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='use BFloat16 as the base datatype for the network')

parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--amsgrad', action='store_true', help='amsgrad for adam')

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help=('0 means no aug, 1 means hard negative mining '
                          'iter 1, 2 means hard negative mining iter 2'))

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=150,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--brt_aug', action='store_true', default=False,
                    help='Use brightness augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--poly_step', type=int, default=110,
                    help='polynomial epoch step')
parser.add_argument('--bs_trn', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=str, default='896',
                    help=('training crop size: either scalar or h,w'))
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--resume', type=str, default=None,
                    help=('continue training from a checkpoint. weights, '
                          'optimizer, schedule are restored'))
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--restore_net', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--result_dir', type=str, default='./logs-dist',
                    help='where to write log output')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help=('Minimum testing to verify nothing failed, '
                          'Runs code for 1 epoch of train and val'))
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
# Full Crop Training
parser.add_argument('--full_crop_training', action='store_true', default=False,
                    help='Full Crop Training')

# Multi Scale Inference
parser.add_argument('--multi_scale_inference', action='store_true',
                    help='Run multi scale inference')

parser.add_argument('--default_scale', type=float, default=1.0,
                    help='default scale to run validation')

parser.add_argument('--log_msinf_to_tb', action='store_true', default=False,
                    help='Log multi-scale Inference to Tensorboard')

parser.add_argument('--eval', type=str, default=None,
                    help=('just run evaluation, can be set to val or trn or '
                          'folder'))
parser.add_argument('--eval_folder', type=str, default=None,
                    help='path to frames to evaluate')
parser.add_argument('--three_scale', action='store_true', default=False)
parser.add_argument('--alt_two_scale', action='store_true', default=False)
parser.add_argument('--do_flip', action='store_true', default=False)
parser.add_argument('--extra_scales', type=str, default='0.5,2.0')
parser.add_argument('--n_scales', type=str, default=None)
parser.add_argument('--align_corners', action='store_true',
                    default=False)
parser.add_argument('--translate_aug_fix', action='store_true', default=False)
parser.add_argument('--mscale_lo_scale', type=float, default=0.5,
                    help='low resolution training scale')
parser.add_argument('--pre_size', type=int, default=None,
                    help=('resize long edge of images to this before'
                          ' augmentation'))
# parser.add_argument('--amp_opt_level', default='O1', type=str,
#                     help=('amp optimization level'))
parser.add_argument('--rand_augment', default=None,
                    help='RandAugment setting: set to \'N,M\'')
parser.add_argument('--init_decoder', default=False, action='store_true',
                    help='initialize decoder with kaiming normal')
parser.add_argument('--dump_topn', type=int, default=0,
                    help='Dump worst val images')
parser.add_argument('--dump_assets', action='store_true',
                    help='Dump interesting assets')
parser.add_argument('--dump_all_images', action='store_true',
                    help='Dump all images, not just a subset')
parser.add_argument('--dump_for_submission', action='store_true',
                    help='Dump assets for submission')
parser.add_argument('--dump_for_auto_labelling', action='store_true',
                    help='Dump assets for autolabelling')
parser.add_argument('--dump_topn_all', action='store_true', default=False,
                    help='dump topN worst failures')
parser.add_argument('--custom_coarse_prob', type=float, default=None,
                    help='Custom Coarse Prob')
parser.add_argument('--only_coarse', action='store_true', default=False)
parser.add_argument('--mask_out_cityscapes', action='store_true',
                    default=False)
parser.add_argument('--ocr_aspp', action='store_true', default=False)
parser.add_argument('--map_crop_val', action='store_true', default=False)
parser.add_argument('--aspp_bot_ch', type=int, default=None)
parser.add_argument('--trial', type=int, default=None)
parser.add_argument('--mscale_cat_scale_flt', action='store_true',
                    default=False)
parser.add_argument('--mscale_dropout', action='store_true',
                    default=False)
parser.add_argument('--mscale_no3x3', action='store_true',
                    default=False, help='no inner 3x3')
parser.add_argument('--mscale_old_arch', action='store_true',
                    default=False, help='use old attention head')
parser.add_argument('--mscale_init', type=float, default=None,
                    help='default attention initialization')
parser.add_argument('--attnscale_bn_head', action='store_true',
                    default=False)
parser.add_argument('--set_cityscapes_root', type=str, default=None,
                    help='override cityscapes default root dir')
parser.add_argument('--ocr_alpha', type=float, default=None,
                    help='set HRNet OCR auxiliary loss weight')
parser.add_argument('--val_freq', type=int, default=1,
                    help='how often (in epochs) to run validation')
parser.add_argument('--deterministic', action='store_true',
                    default=False)
parser.add_argument('--summary', action='store_true',
                    default=False)
parser.add_argument('--segattn_bot_ch', type=int, default=None,
                    help='bottleneck channels for seg and attn heads')
parser.add_argument('--grad_ckpt', action='store_true',
                    default=False)
parser.add_argument('--no_metrics', action='store_true', default=False,
                    help='prevent calculation of metrics')
parser.add_argument('--supervised_mscale_loss_wt', type=float, default=None,
                    help='weighting for the supervised loss')
parser.add_argument('--ocr_aux_loss_rmi', action='store_true', default=False,
                    help='allow rmi for aux loss')

parser.add_argument("--heat", action="store_true", default=True, help="use HeAT")
parser.add_argument("--amp", action="store_true", default=True, help="use torch.cuda.amp")
parser.add_argument(
    "--benchmarking",
    action="store_true",
    default=False,
    help="if true, this will save the loss values and accuracy values to a csv file"
)

args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}


# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
#if args.deterministic:
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

args.heat = False
args.apex = False
args.hvd = True
args.world_size = ht.MPI_WORLD.size
args.rank = ht.MPI_WORLD.rank
args.global_rank = args.rank
rank = args.rank

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        self.avg = torch.tensor(0.)

    def update(self, val, n=1):
        # self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        v = val.detach().cpu() if isinstance(val, torch.Tensor) else val
        self.sum += v  # al.detach().cpu() * n
        self.n += n
        self.avg = self.sum / self.n

def save_checkpoint(state):
    sz = ht.MPI_WORLD.size
    filename = "citys-hvd-checkpoint-" + str(sz) + ".pth.tar"
    torch.save(state, filename)


def print0(*args, **kwargs):
    if ht.MPI_WORLD.rank == 0:
        print(*args, **kwargs)


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def main():
    """
    Main Function
    """
    rank = args.rank
    cfg.GLOBAL_RANK = rank
    args.gpus = torch.cuda.device_count()
    device = torch.device("cpu")
    hvd.init()

    torch.manual_seed(999999)
    #if args.cuda:
    args.cuda = True
        # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    #torch.cuda.manual_seed(args.seed)

    assert args.result_dir is not None, 'need to define result_dir arg'
    logx.initialize(logdir=args.result_dir,
                    tensorboard=True, hparams=vars(args),
                    global_rank=args.global_rank)
    #print("vefore assert and infer")
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args)
    #     args.ngpu = torch.cuda.device_count()
    #     args.best_record = {'mean_iu': -1, 'epoch': 0}
    #print("before datasets / loss")
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    criterion, criterion_val = get_loss(args)

    cwd = os.getcwd()
    sz = ht.MPI_WORLD.size
    filename = cwd + "/citys-hvd-checkpoint-" + str(sz) + ".pth.tar"
    if args.resume and os.path.isfile(filename):
        checkpoint = torch.load(filename,
                                map_location=torch.device('cpu'))
        args.arch = checkpoint['arch']
        args.start_epoch = int(checkpoint['epoch']) + 1
        args.restore_net = True
        args.restore_optimizer = True
        logx.msg(f"Resuming from: checkpoint={args.resume}, " \
                 f"epoch {args.start_epoch}, arch {args.arch}")
    elif args.snapshot:
        if 'ASSETS_PATH' in args.snapshot:
            args.snapshot = args.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
        checkpoint = torch.load(args.snapshot,
                                map_location=torch.device('cpu'))
        args.restore_net = True
        logx.msg(f"Loading weights from: checkpoint={args.snapshot}")

    # todo: HeAT fixes -- urgent -- DDDP / optim / scheduler
    net = network.get_net(args, criterion)
    # net = net.to(device)

    # todo: optim -> direct wrap after this, scheduler stays the same?
    optim, scheduler = get_optimizer(args, net)

    # if args.fp16:
    #     net, optim = amp.initialize(net, optim, opt_level=args.amp_opt_level)
    compression = hvd.Compression.fp16  # if args.fp16_allreduce else hvd.Compression.none

    optim = hvd.DistributedOptimizer(
        optim, named_parameters=net.named_parameters(),
        compression=compression,
        backward_passes_per_step=1,  # args.batches_per_allreduce,
        op=hvd.Average,
        gradient_predivide_factor=1.0,  # args.gradient_predivide_factor)
    )
    #print("after hvd optimizer setup")

    if args.summary:
        print(str(net))
        from thop import profile
        img = torch.randn(1, 3, 1024, 2048).cuda()
        mask = torch.randn(1, 1, 1024, 2048).cuda()
        macs, params = profile(net, inputs={'images': img, 'gts': mask})
        print0(f'macs {macs} params {params}')
        sys.exit()

    if args.restore_optimizer:
        restore_opt(optim, checkpoint)
    if args.restore_net:
        #net.loat_state_dict(checkpoint["state_dict"])
        restore_net(net, checkpoint)

    if args.init_decoder:
        net.module.init_mods()

    torch.cuda.empty_cache()
    #print("before parameter broadcasts")
    #hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    #hvd.broadcast_optimizer_state(optim, root_rank=0)

    if args.start_epoch != 0:
        # TODO: need a loss value for the restart at a certain epoch...
        scheduler.step(args.start_epoch)
    
    #net = net.cuda()
    # There are 4 options for evaluation:
    #  --eval val                           just run validation
    #  --eval val --dump_assets             dump all images and assets
    #  --eval folder                        just dump all basic images
    #  --eval folder --dump_assets          dump all images and assets
    # todo: HeAT fixes -- not urgent --
    # if args.eval == 'val':
    #     if args.dump_topn:
    #         validate_topn(val_loader, net, criterion_val, optim, 0, args)
    #     else:
    #         validate(val_loader, net, criterion=criterion_val, optim=optim, epoch=0,
    #                  dump_assets=args.dump_assets,
    #                  dump_all_images=args.dump_all_images,
    #                  calc_metrics=not args.no_metrics)
    #     return 0
    # elif args.eval == 'folder':
    #     # Using a folder for evaluation means to not calculate metrics
    #     validate(val_loader, net, criterion=None, optim=None, epoch=0,
    #              calc_metrics=False, dump_assets=args.dump_assets,
    #              dump_all_images=True)
    #     return 0
    # elif args.eval is not None:
    #     raise 'unknown eval option {}'.format(args.eval)

    scaler = None #amp.GradScaler()
    args.amp = False #True

    nodes = str(int(hvd.size() / torch.cuda.device_count()))
    cwd = os.getcwd()
    fname = cwd + "/" + nodes + "-hvd-citys-benchmark"
    if args.resume and rank == 0 and os.path.isfile(fname + ".pkl"):
        with open(fname + ".pkl", "rb") as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {
            "epochs": [],
            nodes + "-avg-batch-time": [],
            nodes + "-total-train-time": [],
            nodes + "-train-loss": [],
            nodes + "-val-loss": [],
            nodes + "-val-iou": [],
            nodes + "-val-time": [],
        }
        print0("Output dict:", fname)
    # train_losses, train_btimes, train_ttime = [], [], []
    # val_losses, val_iu, val_ttime = [], [], []

    for epoch in range(args.start_epoch, args.max_epoch):
        # todo: HeAT fixes -- possible conflict between processes
        update_epoch(epoch)

        if args.only_coarse:  # default: false
            train_obj.only_coarse()
            train_obj.build_epoch()
        elif args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.disable_coarse()
                train_obj.build_epoch()
            else:
                train_obj.build_epoch()
        else:
            pass

        ls, bt, btt = train(train_loader, net, optim, epoch, scaler)
        # dp_optim.epoch_loss_logic(ls, loss_globally_averaged=True)

        # if epoch % args.val_freq == 0:
        vls, iu, vtt = validate(val_loader, net, criterion_val, optim, epoch)
        if args.lr_schedule == "plateau":
            scheduler.step(ls)  # val_loss)
        else:
            scheduler.step()

        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": net.state_dict(),
                    "optimizer": optim.state_dict(),
                    # "skip_stable": optim.stability.get_dict()
                }
            )

        out_dict["epochs"].append(epoch)
        out_dict[nodes + "-train-loss"].append(ls)
        out_dict[nodes + "-avg-batch-time"].append(bt)
        out_dict[nodes + "-total-train-time"].append(btt)
        out_dict[nodes + "-val-loss"].append(vls)
        out_dict[nodes + "-val-iou"].append(iu)
        out_dict[nodes + "-val-time"].append(vtt)
        if args.rank == 0:
            save_obj(out_dict, fname)

    if args.rank == 0:
        print("\nRESULTS\n")
        import pandas as pd
        df = pd.DataFrame.from_dict(out_dict).set_index("epochs")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            # more options can be specified also
            print(df)
        if args.benchmarking:
            try:
                fulldf = pd.read_csv(cwd + "/hvd-bench-results.csv")
                fulldf = pd.concat([df, fulldf], axis=1)
            except FileNotFoundError:
                fulldf = df
            fulldf.to_csv(cwd + "/hvd-bench-results.csv")


def lr_warmup(optimizer, epoch, bn, len_epoch, max_lr=0.4):
    if epoch < 5 and bn is not None:
        lr_adjust = (bn + (epoch * len_epoch)) / float(len_epoch * 5.)
    else:
        return

    args.lr = max_lr * lr_adjust
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr


def train(train_loader, net, optim, curr_epoch, scaler):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    return:
    """
    full_bt = time.perf_counter()
    net.train()

    train_main_loss = Metric('train_loss')  # AverageMeter()
    start_time = None
    warmup_iter = 10
    optim.last_batch = len(train_loader) - 1
    btimes = []
    batch_time = time.perf_counter()
    for i, data in enumerate(train_loader):
        lr_warmup(optim, curr_epoch, i, len(train_loader), max_lr=0.4)

        if i <= warmup_iter:
            start_time = time.time()
        # inputs = (bs,3,713,713)
        # gts    = (bs,713,713)
        images, gts, _img_name, scale_float = data
        batch_pixel_size = images.size(0) * images.size(2) * images.size(3)
        images, gts, scale_float = images.cuda(), gts.cuda(), scale_float.cuda()
        inputs = {'images': images, 'gts': gts}
        optim.zero_grad()
        #if args.amp:
        #    with amp.autocast():
        #        main_loss = net(inputs)
        #        log_main_loss = main_loss.clone().detach_()
        #        log_main_loss = hvd.allreduce(log_main_loss)
        #        # log_wait = optim.comm.Iallreduce(MPI.IN_PLACE, log_main_loss, MPI.SUM)
        #        # log_main_loss = log_main_loss / args.world_size
        #    train_main_loss.update(log_main_loss.item(), batch_pixel_size)
        #    scaler.scale(main_loss).backward()
        #else:
        #print0("before net")
        main_loss = net(inputs)
        #main_loss = main_loss.mean()
        # log_main_loss = main_loss.clone().detach_()
        log_main_loss = hvd.allreduce(main_loss.clone(), name=train_main_loss.name)
        train_main_loss.update(log_main_loss, batch_pixel_size)
        #print0("before backward")
        main_loss.backward()

        # the scaler update is within the optim step
        #if args.amp:
        #    scaler.step(optim)
        #    scaler.update()
        #else:
        #print0("before step")
        optim.step()
        #print0("after step")

        if i >= warmup_iter:
            curr_time = time.time()
            batches = i - warmup_iter + 1
            batchtime = (curr_time - start_time) / batches
        else:
            batchtime = 0

        # if log_wait is not None:
        #     log_wait.Wait()
        # log_main_loss = log_main_loss / args.world_size
        # train_main_loss.update(log_main_loss.item(), batch_pixel_size)

        msg = ('[epoch {}], [iter {} / {}], [train main loss {:0.6f}],'
               ' [lr {:0.6f}] [batchtime {:0.3g}]')
        msg = msg.format(
            curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
            optim.param_groups[-1]['lr'], batchtime)
        logx.msg(msg)

        metrics = {'loss': train_main_loss.avg,
                   'lr': optim.param_groups[-1]['lr']}
        curr_iter = curr_epoch * len(train_loader) + i
        logx.metric('train', metrics, curr_iter)

        if i >= 10 and args.test_mode:
            del data, inputs, gts
            return
        btimes.append(time.perf_counter() - batch_time)
        batch_time = time.perf_counter()

    if args.benchmarking:
        #train_loss_tens = torch.tensor(train_main_loss.avg)
        train_loss_tens = hvd.allreduce(train_main_loss.avg.clone(), name=train_main_loss.name)
        # train_loss_tens /= float(optim.comm.size)
        train_main_loss.avg = train_loss_tens.item()

    return train_main_loss.avg, torch.mean(torch.tensor(btimes)), time.perf_counter() - full_bt


def validate(val_loader, net, criterion, optim, epoch,
             calc_metrics=True,
             dump_assets=False,
             dump_all_images=False):
    """
    Run validation for one epoch

    :val_loader: data loader for validation
    :net: the network
    :criterion: loss fn
    :optimizer: optimizer
    :epoch: current epoch
    :calc_metrics: calculate validation score
    :dump_assets: dump attention prediction(s) images
    :dump_all_images: dump all images, not just N
    """
    val_time = time.perf_counter()
    dumper = ImageDumper(
        val_len=len(val_loader),
        dump_all_images=dump_all_images,
        dump_assets=dump_assets,
        dump_for_auto_labelling=args.dump_for_auto_labelling,
        dump_for_submission=args.dump_for_submission,
        rank=rank
    )

    net.eval()
    val_loss = Metric('vald_loss')  # AverageMeter()
    iou_acc = 0
    for val_idx, data in enumerate(val_loader):
        input_images, labels, img_names, _ = data
        if args.dump_for_auto_labelling or args.dump_for_submission:
            submit_fn = '{}.png'.format(img_names[0])
            if val_idx % 20 == 0:
                logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')
            if os.path.exists(os.path.join(dumper.save_dir, submit_fn)):
                continue

        # Run network
        assets, _iou_acc = \
            eval_minibatch(data, net, criterion, val_loss, calc_metrics,
                          args, val_idx)

        iou_acc += _iou_acc

        input_images, labels, img_names, _ = data

        if hvd.rank() == 0:
            dumper.dump({'gt_images': labels,
                         'input_images': input_images,
                         'img_names': img_names,
                         'assets': assets}, val_idx)

        if val_idx > 5 and args.test_mode:
            break

        if val_idx % 2 == 0 and hvd.rank() == 0:
            logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')

    # average the loss value
    val_loss_tens = val_loss.avg.clone()
    val_loss_tens = hvd.allreduce(val_loss_tens.clone(), name=val_loss.name)
    val_loss.avg = val_loss_tens.item()
    # sum up the iou_acc
    iou_acc = hvd.allreduce(torch.tensor(iou_acc), name=val_loss.name) * hvd.size()

    # was_best = False
    if calc_metrics:
        # was_best = eval_metrics(iou_acc, args, net, optim, val_loss, epoch)
        _, mean_iu = eval_metrics(iou_acc, args, net, optim, val_loss, epoch)

    # optim.comm.bcast(mean_iu, root=0)
    if hvd.rank() != 0:
        mean_iu = None
    # was_best = optim.comm.bcast(was_best, root=0)
    #
    # # Write out a summary html page and tensorboard image table
    # if not args.dump_for_auto_labelling and not args.dump_for_submission and optim.comm.rank == 0:
    #     dumper.write_summaries(was_best)
    return val_loss.avg, mean_iu, time.perf_counter() - val_time


if __name__ == '__main__':
    main()
