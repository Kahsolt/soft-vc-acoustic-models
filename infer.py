#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/12 

import os
from argparse import ArgumentParser
from traceback import print_exc

import torch
import torchaudio
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from scipy.io import wavfile

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_PATH = 'log'


def convert(args):
  hubert   = torch.hub.load("bshall/hubert:main",          "hubert_soft").to(device)
  acoustic = torch.hub.load("bshall/acoustic-model:main",  "hubert_soft").to(device)
  hifigan  = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").to(device)

  if args.vbank == '*':
    vbanks = os.listdir(CKPT_PATH)
  else:
    vbanks = [args.vbank]
  
  for vbank in vbanks:
    print(f'>> [Generate] for vbank {vbank}')

    if vbank != 'default':
      ckpt = torch.load(f'{CKPT_PATH}/{vbank}/model-best.pt', map_location=device)
      consume_prefix_in_state_dict_if_present(ckpt["acoustic-model"], "module.")
      acoustic.load_state_dict(ckpt["acoustic-model"])

    if os.path.isfile(args.input):
      wav_fps = [args.input]
    else:
      wav_fps = [os.path.join(args.input, fn) for fn in os.listdir(args.input)]
    os.makedirs(args.out_path, exist_ok=True)
    
    SAMPLE_RATE = 16000
    with torch.inference_mode():
      for wav_fp in wav_fps:
        try:
          source, sr = torchaudio.load(wav_fp)
          source = torchaudio.functional.resample(source, sr, SAMPLE_RATE)
          source = source.unsqueeze(0).to(device)
          
          units = hubert.units(source)
          mel = acoustic.generate(units).transpose(1, 2)
          target = hifigan(mel)

          y_hat = target.squeeze().cpu().numpy()
          name, ext = os.path.splitext(os.path.basename(wav_fp))
          save_fp = os.path.join(args.out_path, f'{name}_{vbank}{ext}')
          wavfile.write(save_fp, SAMPLE_RATE, y_hat)
          print(f'>> {save_fp}')
        except Exception as e:
          print_exc()
          #print(f'<< [Error] {e}')
          print(f'<< ignore file {wav_fp}')


if __name__ == '__main__':
  VBANKS = os.listdir(CKPT_PATH) + ['*', 'default']   # where ckpt locates

  parser = ArgumentParser()
  parser.add_argument("vbank", metavar='vbank', default='*', choices=VBANKS, help='voice bank name; use `*` for all local, use `default` for original pretrained weights from soft-vc repo')
  parser.add_argument("--input", default='test', help='input file or folder for conversion')
  parser.add_argument("--out_path", default='gen', help='output folder for converted wavfiles')
  args = parser.parse_args()

  convert(args)
