import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer


def main(checkpoint):
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()


if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    main(checkpoint)
