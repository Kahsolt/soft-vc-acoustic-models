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

  print(f"Encoding dataset at {args.wav_path}")
  with torch.inference_mode():
    for wav_fp in tqdm(list(args.wav_path.rglob("*.wav"))):
      wav, sr = torchaudio.load(wav_fp)
      wav = resample(wav, sr, SAMPLE_RATE)
      wav = wav.unsqueeze(0).to(device)

      units = hubert.units(wav)

      out_path = args.out_path / wav_fp.relative_to(args.wav_path)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


def process_wav(wav_fp, out_path):
  wav, sr = torchaudio.load(wav_fp)
  wav = resample(wav, sr, SAMPLE_RATE)
  wav = wav.unsqueeze(0)

  logmel = melspectrogram(wav)

  np.save(out_path.with_suffix(".npy"), logmel.squeeze().numpy())
  return out_path, logmel.shape[-1]


def preprocess_dataset(args):
  args.out_path.mkdir(parents=True, exist_ok=True)

  futures = []
  executor = ProcessPoolExecutor(max_workers=cpu_count())
  print(f"Extracting features for {args.wav_path}")
  for wav_fp in args.wav_path.rglob("*.wav"):
    relative_path = wav_fp.relative_to(args.wav_path)
    out_path = args.out_path / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    futures.append(executor.submit(process_wav, wav_fp, out_path))

  results = [future.result() for future in tqdm(futures)]

  lengths = {path.stem: length for path, length in results}
  frames = sum(lengths.values())
  hours = frames * (160 / SAMPLE_RATE) / 3600
  print(f"Found {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")


if __name__ == "__main__":
  VBANKS = os.listdir('data')   # where train data locates

  parser = ArgumentParser()
  parser.add_argument("vbank", metavar='vbank', choices=VBANKS, help='voice bank name')
  parser.add_argument("--encode", action='store_true', help='generated HuBERT hidden-units')
  parser.add_argument("--melspec", action='store_true', help='generated logscale-melspec')
  args = parser.parse_args()

  args.wav_path = Path('data') / args.vbank / 'wavs'

  if args.encode:
    args.out_path = Path('data') / args.vbank / 'units'
    encode_dataset(args)
  elif args.melspec:
    args.out_path = Path('data') / args.vbank / 'mels'
    preprocess_dataset(args)
  else:
    raise('either --encode, --melspec must be set')
