#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/26 

# utils about ckpt files

import os
from argparse import ArgumentParser

LOG_PATH = 'log'


def clean(args):
  for dn in os.listdir(LOG_PATH):
    dp = os.path.join(LOG_PATH, dn)
    for fn in os.listdir(dp):
      fp = os.path.join(dp, fn)
      if not os.path.isfile(fp): continue
      if fn == 'model-best.pt': continue

      if args.f:
        print(f'>> removing {fp}')
        os.unlink(fp)
      else:
        print(f'>> will remove {fp}')


def show(args):
  import torch

  for dn in os.listdir(LOG_PATH):
    fp = os.path.join(LOG_PATH, dn, 'model-best.pt')
    ckpt = torch.load(fp)
    print(dn, ckpt['step'], ckpt['loss'])


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--clean', action='store_true', help='clean all midway ckpt files, use with -f')
  parser.add_argument('-f', action='store_true', help='actually do file remove')
  parser.add_argument('--show')
  args = parser.parse_args()

  if args.clean: clean(args)
  if args.show:  show(args)
