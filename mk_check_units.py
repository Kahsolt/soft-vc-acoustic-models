#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/26 

# NOTE: run this before mk_preprocess.cmd
# to comprehensively check how your dataset covers the 100 classes (cases) of the pretrained HuBERT

import os
from pathlib import Path
from collections import Counter
from argparse import ArgumentParser

import torch
import torchaudio
from torchaudio.functional import resample
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# NOTE: this is fixed in the pretrained hubert
SAMPLE_RATE = 16000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_units(args):
  print(f"Loading pretrained hubert checkpoint")
  hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(device)

  print(f"Checking dataset at {args.wav_path}")
  preds = []
  with torch.inference_mode():
    for wav_fp in tqdm(list(args.wav_path.rglob("*.wav"))):
      wav, sr = torchaudio.load(wav_fp)
      wav = resample(wav, sr, SAMPLE_RATE)
      wav = wav.unsqueeze(0).to(device)

      logits, _ = hubert(wav)
      pred = logits.argmax(dim=-1)
      preds.append(pred.squeeze())

  preds = torch.cat(preds, dim=0)
  preds = preds.cpu().numpy().tolist()

  if 'hist':
    plt.hist(preds, bins=100)
    plt.savefig(args.img_path / f'units_ditribution_{args.vbank}.png')
    plt.show()

  cntr = Counter(preds)
  freq = sorted(cntr.values(), reverse=True)

  if 'sorted freq':
    plt.subplot(211) ; plt.title('sorted freq')       ; plt.plot(freq)
    plt.subplot(212) ; plt.title('sorted freq (log)') ; plt.plot(np.log(freq))
    plt.savefig(args.img_path / f'units_sorted_freq_{args.vbank}.png')
    plt.show()


if __name__ == '__main__':
  VBANKS = os.listdir('data')   # where train data locates

  parser = ArgumentParser()
  parser.add_argument("vbank", metavar='vbank', choices=VBANKS, help='voice bank name')
  parser.add_argument("--img_path", type=Path, default='img', help='savefig img folder')
  args = parser.parse_args()

  args.wav_path = Path('data') / args.vbank / 'wavs'

  os.makedirs(args.img_path, exist_ok=True)
  check_units(args)
