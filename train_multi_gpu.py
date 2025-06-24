import torch
import torchaudio
import gc
import argparse
import os
from tqdm import tqdm
import wandb
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler, StyleVDiffusion, StyleVSampler
from diffusers import AudioLDMPipeline
from audio_data_pytorch import AllTransform
from ss_speech_dataset import SoundSpacesSpeechDataset
import soundfile as sf
import random
import laion_clap
import pdb

from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
from facodec_diffusion import FACodecDiffusion
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

SAMPLE_RATE = 16000
BATCH_SIZE = 12
NUM_SAMPLES = int(2.56 * SAMPLE_RATE)

def main():
    args = parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FACodecDiffusion(args)

    print('Load data')
    dataset = SoundSpacesSpeechDataset
    train_dataset = dataset(split='train', use_rgb=True, use_depth=True, hop_length=128,
                        remove_oov=True, use_librispeech=False,
                        convolve_random_rir=False, use_da=False,
                        read_mp4=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=2, pin_memory=True,
                                                batch_size=BATCH_SIZE)
    valid_dataset = dataset(split='val', use_rgb=True, use_depth=True, hop_length=128,
                        remove_oov=True, convolve_random_rir=False, use_da=False,
                        read_mp4=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=2, batch_size=BATCH_SIZE, pin_memory=True)

    print('- Number of training samples: {}'.format(len(train_dataset)))
    print('- Number of validation samples: {}'.format(len(valid_dataset)))

    # Model training
    logging_name = f"{args.model_dir}_{args.version}"
    logger = loggers.TensorBoardLogger(save_dir = f"./logs/", name = logging_name)

	# Callbacks
    callbacks = []
    callbacks.append(EarlyStopping(monitor="valid_loss", mode="min", patience=50))
    callbacks.append(ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor="fd_score",
        mode="min",
        filename="facodec_{epoch:04d}_{fd_score:.4f}",
        #every_n_val_epochs=args.ckpt_interval,
        save_top_k=1,
        verbose=True,
    ))
    callbacks.append(ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor="valid_loss",
        mode="min",
        filename="facodec_{epoch:04d}_{valid_loss:.4f}",
        #every_n_val_epochs=args.ckpt_interval,
        save_top_k=1,
        verbose=True,
        save_last=True,
    ))

    trainer = Trainer(
        devices=args.n_gpus,
        num_nodes=args.num_node,
        #strategy="ddp",
        accelerator="gpu",
        benchmark=True,
        max_epochs=args.max_epochs,
        #val_check_interval=0.001,
        strategy=DDPStrategy(find_unused_parameters=True),
        #resume_from_checkpoint=args.from_pretrained,
        default_root_dir=args.model_dir,
        callbacks=callbacks,
        logger=logger,
        #progress_bar_refresh_rate=args.fast_dev_run or args.progress_bar,
        fast_dev_run=args.fast_dev_run,
        #plugins=DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(model, train_loader, valid_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='/home/pantianrui/av_ldm/history/')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", type=str) # , default='ldm_audio_condition_no_overlap'
    parser.add_argument("--n-gpus", default=1, type=int)
    parser.add_argument("--num-node", default=1, type=int)
    parser.add_argument("--max-epochs", default=2500, type=int)
    parser.add_argument("--from-pretrained", default='./', type=str)
    parser.add_argument("--model-dir", default='avldm')
    parser.add_argument("--version", default='v1')
    parser.add_argument("--fast-dev-run", default=False, action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    main()

