import os
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from acoustic import AcousticModel
from acoustic.dataset import MelDataset
from acoustic.utils import Metric, save_checkpoint, load_checkpoint, plot_spectrogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(rank, world_size, args, hp):
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    dist.init_process_group(hp.BACKEND, rank=rank, world_size=world_size, init_method=hp.INIT_METHOD)

    ####################################################################################
    # Setup logging utilities:
    ####################################################################################

    log_dir = args.log_path / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir / f"{args.log_path.stem}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    ####################################################################################
    # Initialize models and optimizer
    ####################################################################################

    acoustic = AcousticModel().to(rank)
    acoustic = DDP(acoustic, device_ids=[rank])

    optimizer = optim.AdamW(
        acoustic.parameters(),
        lr=hp.LEARNING_RATE,
        betas=hp.BETAS,
        weight_decay=hp.WEIGHT_DECAY,
    )

    ####################################################################################
    # Initialize datasets and dataloaders
    ####################################################################################

    train_dataset = MelDataset(
        root=args.data_path,
        train=True,
        split_ratio=args.split_ratio,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=train_dataset.pad_collate,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    validation_dataset = MelDataset(
        root=args.data_path,
        train=False,
        split_ratio=args.split_ratio,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    ####################################################################################
    # Load checkpoint if args.resume is set
    ####################################################################################

    if args.resume:
        global_step, best_loss = load_checkpoint(
            load_path=args.resume,
            acoustic=acoustic,
            optimizer=optimizer,
            rank=rank,
            logger=logger,
        )
    else:
        global_step, best_loss = 0, float("inf")

    # =================================================================================#
    # Start training loop
    # =================================================================================#

    n_epochs = hp.STEPS // len(train_loader) + 1
    start_epoch = global_step // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
        logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
        logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")
    logger.info(f"batch size: {hp.BATCH_SIZE}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"# of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info(f"")
    logger.info(f'BATCH_SIZE: {hp.BATCH_SIZE}')
    logger.info(f'LEARNING_RATE: {hp.LEARNING_RATE}')
    logger.info(f'BETAS: {hp.BETAS}')
    logger.info(f'WEIGHT_DECAY: {hp.WEIGHT_DECAY}')
    logger.info(f'STEPS: {hp.STEPS}')
    logger.info(f'LOG_INTERVAL: {hp.LOG_INTERVAL}')
    logger.info(f'VALIDATION_INTERVAL: {hp.VALIDATION_INTERVAL}')
    logger.info(f'CHECKPOINT_INTERVAL: {hp.CHECKPOINT_INTERVAL}')
    logger.info(f'BACKEND: {hp.BACKEND}')
    logger.info(f'INIT_METHOD: {hp.INIT_METHOD}')
    logger.info("**" * 40 + "\n")

    average_loss    = Metric()
    epoch_loss      = Metric()
    validation_loss = Metric()

    acoustic.train()
    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)
        epoch_loss.reset()

        for mels, mels_lengths, units, units_lengths in train_loader:
            mels,  mels_lengths  = mels .to(rank), mels_lengths .to(rank)
            units, units_lengths = units.to(rank), units_lengths.to(rank)

            ############################################################################
            # Compute training loss
            ############################################################################

            optimizer.zero_grad()

            mels_ = acoustic(units, mels[:, :-1, :])

            loss = F.l1_loss(mels_, mels[:, 1:, :], reduction="none")
            loss = torch.sum(loss, dim=(1, 2)) / (mels_.size(-1) * mels_lengths)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            global_step += 1

            ############################################################################
            # Update and log training metrics
            ############################################################################

            average_loss.update(loss.item())
            epoch_loss  .update(loss.item())

            if rank == 0 and global_step % hp.LOG_INTERVAL == 0:
                logger.info(f">> [Step {global_step}] loss: {average_loss.value}")
                writer.add_scalar("train/loss", average_loss.value, global_step)
                average_loss.reset()

            # --------------------------------------------------------------------------#
            # Start validation loop
            # --------------------------------------------------------------------------#

            if global_step % hp.VALIDATION_INTERVAL == 0:
                acoustic.eval()
                validation_loss.reset()

                for i, (mels, units) in enumerate(validation_loader, 1):
                    mels, units = mels.to(rank), units.to(rank)

                    with torch.no_grad():
                        mels_ = acoustic(units, mels[:, :-1, :])
                        loss = F.l1_loss(mels_, mels[:, 1:, :])

                    ####################################################################
                    # Update validation metrics and log generated mels
                    ####################################################################

                    validation_loss.update(loss.item())

                    if rank == 0 and i < 4:     # display first three samples
                        if global_step == hp.VALIDATION_INTERVAL:     # if the firts time
                            writer.add_figure(f"original/mel_{i}", plot_spectrogram(mels .squeeze().transpose(0, 1).cpu().numpy()), global_step)
                        writer.add_figure(f"generated/mel_{i}", plot_spectrogram(mels_.squeeze().transpose(0, 1).cpu().numpy()), global_step)

                acoustic.train()

                ############################################################################
                # Log validation metrics
                ############################

                if rank == 0:
                    writer.add_scalar("validation/loss", validation_loss.value, global_step)
                    logger.info(f"valid -- epoch: {epoch}, loss: {validation_loss.value:.4f}")

                new_best = best_loss > validation_loss.value
                if new_best or global_step % hp.CHECKPOINT_INTERVAL:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = validation_loss.value

                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir=args.log_path,
                            acoustic=acoustic,
                            optimizer=optimizer,
                            step=global_step,
                            loss=validation_loss.value,
                            best=new_best,
                            logger=logger,
                        )

            # -----------------------------------------------------------------------------#
            # End validation loop
            # -----------------------------------------------------------------------------#

        ####################################################################################
        # Log training metrics
        ####################################################################################

        logger.info(f"train -- epoch: {epoch}, loss: {epoch_loss.value:.4f}")

        # =================================================================================#
        # End training loop
        # ==================================================================================#

    dist.destroy_process_group()


if __name__ == "__main__":
    VBANKS = os.listdir('data')   # where train data locates

    parser = ArgumentParser()
    parser.add_argument("vbank", metavar='vbank', choices=VBANKS, help='voice bank name')
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", help="path to the checkpoint file to resume from", type=Path)
    parser.add_argument("--split_ratio", default=0.1, help="dataset valid/train split ratio")
    args = parser.parse_args()

    args.data_path = Path('data') / args.vbank
    args.log_path  = Path('log')  / args.vbank

    with open(args.config, 'r', encoding='utf-8') as fh:
        config = json.load(fh)

    # NOTE: these defaults may be overwritten by `config.json`
    # borrow this seralizable `args` to hold hparams
    hp = args
    hp.BATCH_SIZE          = config.get("batch_size",          32         )
    hp.LEARNING_RATE       = config.get("learning_rate",       4e-4       )
    hp.BETAS               = config.get("betas",               (0.8, 0.99))
    hp.WEIGHT_DECAY        = config.get("weight_decay",        1e-5       )
    hp.STEPS               = config.get("steps",               80000      )
    hp.LOG_INTERVAL        = config.get("log_interval",        5          )
    hp.VALIDATION_INTERVAL = config.get("validation_interval", 1000       )
    hp.CHECKPOINT_INTERVAL = config.get("checkpoint_interval", 1000       )
    #args.BACKEND             = 'nccl'           # use this on Linux
    hp.BACKEND             = 'gloo'           # use this on Windows
    hp.INIT_METHOD         = 'tcp://localhost:54321'

    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args, hp), nprocs=world_size, join=True)
