content = open('e:/project/FedAF/main_feddm_svd_cnn_new_4_1.py', encoding='utf-8', errors='replace').read()

start_marker = "                image_syn = torch.randn(size=(len(classes)*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,  requires_grad=True, device=args.device)\n                # image_syn = user.image_syn"
end_marker = "                exp_user_path = os.path.join(args.vis_path"

start = content.find(start_marker)
end = content.find(end_marker, start)
print('start:', start, 'end:', end)
assert start != -1 and end != -1, "markers not found"

new_block = """                # ---- warm start: initialise from real images in round 0; reuse previous round result thereafter ----
                if curr_epoch == 0:
                    image_syn = torch.randn(size=(len(classes)*args.ipc, channel, im_size[0], im_size[1]),
                                            dtype=torch.float, requires_grad=True, device=args.device)
                    if args.init == 'real':
                        logging.info('initialize synthetic data from random real images with pseudo labels')
                        if args.dataset == 'ImageNet':
                            image_syn.data.copy_(imnet_init_images.data)
                        else:
                            for i, c in enumerate(classes):
                                if not args.aug:
                                    image_syn.data[i*args.ipc:(i+1)*args.ipc] = get_images(c, args.ipc).detach().data
                                else:
                                    half_size = im_size[0]//2
                                    image_syn.data[i*args.ipc:(i+1)*args.ipc, :, :half_size, :half_size] = downscale(get_images(c, args.ipc), 0.5).detach().data
                                    image_syn.data[i*args.ipc:(i+1)*args.ipc, :, half_size:, :half_size] = downscale(get_images(c, args.ipc), 0.5).detach().data
                                    image_syn.data[i*args.ipc:(i+1)*args.ipc, :, :half_size, half_size:] = downscale(get_images(c, args.ipc), 0.5).detach().data
                                    image_syn.data[i*args.ipc:(i+1)*args.ipc, :, half_size:, half_size:] = downscale(get_images(c, args.ipc), 0.5).detach().data
                    elif args.init == 'pretrained':
                        logging.info('initialize synthetic data from pretrained images')
                        ckpt_exp_user_path = os.path.join(args.ckpt_path, 'exp_{}'.format(exp), 'user_{}'.format(idx))
                        if not os.path.exists(ckpt_exp_user_path):
                            os.makedirs(ckpt_exp_user_path)
                        data_path = os.path.join(ckpt_exp_user_path, 'run_%s_%s_%d.pt'%(args.dataset, args.model, curr_epoch-1))
                        syn_state = torch.load(data_path)
                        assert syn_state['data'][0].shape[0] == args.ipc * len(classes)
                        image_syn.data.copy_(syn_state['data'][0].to(args.device))
                        label_syn.data.copy_(syn_state['data'][1])
                    else:
                        logging.info('initialize synthetic data from random noise for user %d'%idx)
                        img_real = user.dataset.get_random_images(args.batch_real).detach().data
                        image_syn.requires_grad_(False)
                        image_syn[:,0,:,:] = image_syn[:,0,:,:] / image_syn[:,0,:,:].abs().max() * img_real[:,0,:,:].abs().max()
                else:
                    # warm start from previous round
                    image_syn = user.image_syn.detach().clone().to(args.device).requires_grad_(True)

                mask = torch.zeros(size=(len(classes) * args.ipc, channel, im_size[0], im_size[1]),
                                        dtype=torch.float, device=args.device)
                height, width = 32, 32
                mask[:, :, :height, :width] = 1.0
                mask.requires_grad = False

                optimizer_img = get_optimizer([image_syn, ], args.opt_X, lr=args.lr_img, weight_decay=0, rho=0, momentum=0.5)
                optimizer_img.zero_grad()

                # ---- fix one feature extractor per round; no longer swap networks every iteration ----
                if curr_epoch != 0:
                    iter_net = random_perturb(copy.deepcopy(global_model))
                else:
                    iter_net = get_network(args.model, channel, num_classes, im_size).cuda()
                iter_net.train()
                for param in iter_net.parameters():
                    param.requires_grad = False
                embed = iter_net.module.embed if torch.cuda.device_count() > 1 else iter_net.embed

                for it in range(args.Iteration):
                    progress = (it + 1) / args.Iteration
                    total_loss = torch.tensor(0.0, device=args.device)
                    labs_syn = torch.zeros(0, dtype=torch.long, device=args.device)

                    for i, c in enumerate(classes):
                        img_real = get_images(c, args.batch_real)
                        img_syn_raw = image_syn[i * args.ipc:(i + 1) * args.ipc].reshape(
                            (args.ipc, channel, im_size[0], im_size[1]))
                        lab_c = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                        # subset for redundancy / synergy
                        V_raw = sample_subset(img_syn_raw.detach(), ratio=0.5)
                        lab_sub = torch.ones((V_raw.shape[0],), device=args.device, dtype=torch.long) * c

                        # aug versions
                        if args.aug:
                            img_syn_all, lab_all = number_sign_augment(img_syn_raw, lab_c)
                            img_syn_sub, lab_sub_aug = number_sign_augment(V_raw, lab_sub)
                        else:
                            img_syn_all, lab_all = img_syn_raw, lab_c
                            img_syn_sub, lab_sub_aug = V_raw, lab_sub

                        # two independent DSA views of raw syn for uniqueness consistency
                        if args.dsa:
                            seed  = int(time.time() * 1000) % 100000
                            seed_a = int(time.time() * 1001) % 100000
                            seed_b = int(time.time() * 1003) % 100000
                            img_real    = DiffAugment(img_real,    args.dsa_strategy, seed=seed,   param=args.dsa_param)
                            img_syn_all = DiffAugment(img_syn_all, args.dsa_strategy, seed=seed,   param=args.dsa_param)
                            img_syn_sub = DiffAugment(img_syn_sub, args.dsa_strategy, seed=seed,   param=args.dsa_param)
                            img_syn_a   = DiffAugment(img_syn_raw, args.dsa_strategy, seed=seed_a, param=args.dsa_param)
                            img_syn_b   = DiffAugment(img_syn_raw, args.dsa_strategy, seed=seed_b, param=args.dsa_param)
                        else:
                            img_syn_a = img_syn_raw
                            img_syn_b = img_syn_raw

                        _, feat_real,    logit_real = embed(img_real)
                        _, feat_syn_all, logit_syn  = embed(img_syn_all)
                        _, feat_syn_sub, logit_sub  = embed(img_syn_sub)
                        _, feat_syn_a,   _           = embed(img_syn_a)
                        _, feat_syn_b,   _           = embed(img_syn_b)
                        feat_real_d = feat_real.detach()

                        # A. Redundancy loss (primary task)
                        L_R = loss_redundancy(feat_real_d, feat_syn_all, logit_syn,
                                              feat_syn_sub, logit_sub, lab_all, lab_sub_aug)

                        # B. Synergy loss (stage 2+)
                        if progress >= 0.3:
                            L_S = loss_synergy(feat_real_d, feat_syn_all, feat_syn_sub)

                        # C. Uniqueness loss (stage 3)
                        if progress >= 0.8:
                            L_U = loss_uniqueness(feat_syn_a, feat_syn_b, feat_real_d)

                        # stage-wise combination
                        if progress < 0.3:
                            loss_c = L_R
                        elif progress < 0.8:
                            loss_c = L_R + lambda_S * L_S
                        else:
                            loss_c = L_R + lambda_S * L_S + lambda_U * L_U

                        total_loss += loss_c
                        labs_syn = torch.cat([labs_syn, lab_all], dim=0)

                    optimizer_img.zero_grad()
                    total_loss.backward()
                    optimizer_img.step()

                    if (it + 1) % 500 == 0:
                        logging.info('%s user=%d epoch=%d iter=%d loss=%.4f' % (
                            get_time(), idx, curr_epoch, it + 1, total_loss.item()))

                # warm start: save current round result for use in the next round
                user.image_syn = image_syn.detach().clone()

                """

new_content = content[:start] + new_block + content[end:]
open('e:/project/FedAF/main_feddm_svd_cnn_new_4_1.py', 'w', encoding='utf-8').write(new_content)
print('Done, new length:', len(new_content))
