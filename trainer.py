from decimal import Decimal
import utility
import gc
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import pytorch_colors as colors
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use('TkAgg')


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        # self.args.cpu = False

        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.Timer(), utility.Timer()

        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):

        # self.args.cpu = True

        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.zero_grad()
        self.model.eval()

        timer_test = utility.Timer()
        if self.args.save_results:
            self.ckp.begin_background()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                ssim_total = 0
                psnr = 0
                d.dataset.set_scale(idx_scale)

                for lr, hr, filename in tqdm(d, ncols=80):
                    torch.cuda.empty_cache()
                    gc.collect()

                    lr, hr = self.prepare(lr, hr)

                    sr = self.model(lr, idx_scale)

                    if self.args.use_lab:
                        hr = colors.lab_to_rgb(hr) * self.args.rgb_range
                        lr = colors.lab_to_rgb(lr) * self.args.rgb_range
                        sr = colors.lab_to_rgb(sr) * self.args.rgb_range

                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    psnr += utility.calc_psnr( sr, hr, scale, self.args.rgb_range, dataset=d)
                    ssim_total += utility.calc_ssim(sr, hr, self.args.rgb_range).item()

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                psnr /= len(d)
                best = self.ckp.log.max(0)
                """"
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                """
                print(psnr)
                ssim_val = ssim_total / len(d)
                self.ckp.write_log(
                    '[{} x{}]\tSSIM: {:.3f}'.format(
                        d.dataset.name,
                        scale,
                        ssim_val
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def get_feature_maps(self):
        model_children = self.model.model.children()
        children = list(model_children)
        body_block_ind = 3
        body_children = children[3][body_block_ind].children()
        activation_children = list(body_children)
        #for child in body_children:
        #    print(child)
        asca_layer = list(activation_children[0][0].children())[1]
        mask_layer = list(activation_children[0][0].children())[4]
        res_blocks_after_mixed_attention = list(list(activation_children[0][0].children())[5].children())[0]

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        sample_index = 4
        test_dataset_index = 0
        feature_map_index = 4
        layer_name_to_save = "final_attention_map"
        res_blocks_after_mixed_attention.register_forward_hook(get_activation("final_attention_map"))
        asca_layer.register_forward_hook(get_activation("asca_layer"))
        mask_layer.register_forward_hook(get_activation("mask_layer"))

        dataset = self.loader_test[test_dataset_index].dataset
        lr_input, gt, filename = dataset[sample_index]
        lr_input = lr_input.cuda(non_blocking=True)
        lr_input.unsqueeze_(0)
        output = self.model.model(lr_input)

        act = activation[layer_name_to_save].squeeze()

        size = act.size(0)
        # size = 2
        """"
        fig, axarr = plt.subplots(size)

        for idx in range(size):
            plt.imshow(act[idx].cpu().numpy())
            # plt.show()
        """
        save_folder = os.path.join(self.ckp.dir, "feature_maps-"+layer_name_to_save+"-"+dataset.name)
        os.makedirs(os.path.join(save_folder), exist_ok=True)
        for idx in range(size):
            # plt.figure()
            save_name = filename + "_" + str(idx)
            save_dir = os.path.join(save_folder, save_name)

            feature_act = act[idx]
            # feature_act = torch.nn.functional.softmax(feature_act) burasi acilabilir TODO

            plt.imshow(feature_act.cpu().numpy()) # TODO cmap='hot', interpolation='nearest'
            plt.savefig(save_dir)
            # plt.show()
        # plt.imshow(act[feature_map_index].cpu().numpy())
        # plt.show()

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

