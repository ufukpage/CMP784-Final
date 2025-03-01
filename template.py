def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('RNAN') >= 0:
        args.model = 'RNAN'
        # args.n_resgroups = 16
        args.n_feats = 64
        args.chop = True

    if args.template.find('EARN') >= 0:
        args.model = 'EARN'
        # args.n_resgroups = 10
        args.n_feats = 64
        args.chop = True

    if args.template.find('EARN_NOASCA') >= 0:
        args.model = 'EARN_NOASCA'
        # args.n_resgroups = 10
        args.n_feats = 64
        args.chop = True

    if args.template.find('EARN_NOLA') >= 0:
        args.model = 'EARN_NOLA'
        # args.n_resgroups = 10
        args.n_feats = 64
        args.chop = True

    if args.template.find('EARN_NOSKIP') >= 0:
        args.model = 'EARN_NOSKIP'
        # args.n_resgroups = 10
        args.n_feats = 64
        args.chop = True

    if args.template.find('SAN') >= 0:
        args.model = 'SAN'
        args.n_resblocks = 10
        args.n_resgroups = 20
        args.resduction = 16
        args.n_feats = 64
        args.chop = True

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True


