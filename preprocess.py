import os
from pathlib import Path
from multiprocessing import cpu_count
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm
import torch
import torchaudio
from torchaudio.functional import resample

from acoustic.utils import LogMelSpectrogram

# NOTE: this is fixed in the pretrained hubert
SAMPLE_RATE = 16000
melspectrogram = LogMelSpectrogram()


def encode_dataset(args):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  print(f"Loading pretrained hubert checkpoint")
  hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(device)

  print(f"Encoding dataset at {args.in_dir}")
  with torch.inference_mode():
    for in_path in tqdm(list(args.in_dir.rglob("*.wav"))):
      wav, sr = torchaudio.load(in_path)
      wav = resample(wav, sr, SAMPLE_RATE)
      wav = wav.unsqueeze(0).to(device)

      units = hubert.units(wav)

      out_path = args.out_dir / in_path.relative_to(args.in_dir)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


def process_wav(in_path, out_path):
  wav, sr = torchaudio.load(in_path)
  wav = resample(wav, sr, SAMPLE_RATE)
  wav = wav.unsqueeze(0)

  logmel = melspectrogram(wav)

  np.save(out_path.with_suffix(".npy"), logmel.squeeze().numpy())
  return out_path, logmel.shape[-1]


def preprocess_dataset(args):
  args.out_dir.mkdir(parents=True, exist_ok=True)

  futures = []
  executor = ProcessPoolExecutor(max_workers=cpu_count())
  print(f"Extracting features for {args.in_dir}")
  for in_path in args.in_dir.rglob("*.wav"):
    relative_path = in_path.relative_to(args.in_dir)
    out_path = args.out_dir / relative_path.with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    futures.append(executor.submit(process_wav, in_path, out_path))

  results = [future.result() for future in tqdm(futures)]

  lengths = {path.stem: length for path, length in results}
  frames = sum(lengths.values())
  hours = frames * (160 / SAMPLE_RATE) / 3600
  print(f"Found {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")


def make_mix_dataset(args):
  vbanks = os.listdir('data')
  print(vbanks)

  out_dp = os.path.join('data', 'mix')
  os.makedirs(out_dp, exist_ok=True)
  units_dp = os.path.join(out_dp, 'units')
  os.makedirs(units_dp, exist_ok=True)
  mels_dp = os.path.join(out_dp, 'mels')
  os.makedirs(mels_dp, exist_ok=True)

  raise NotImplementedError


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--encode", action='store_true')
  parser.add_argument("--melspec", action='store_true')
  parser.add_argument("--make_mix", action='store_true')
  parser.add_argument("in_dir", metavar="in-dir", help="path to the dataset directory", type=Path)
  parser.add_argument("out_dir", metavar="out-dir", help="path to the output directory", type=Path)
  args = parser.parse_args()

  if   args.encode:      encode_dataset(args)
  elif args.melspec: preprocess_dataset(args)
  elif args.make_mix:  make_mix_dataset(args)
  else: raise('either --encode, --melspec or --make_mix must be set')
