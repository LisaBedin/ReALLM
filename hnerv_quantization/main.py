import hydra
import os
from omegaconf import DictConfig, OmegaConf
import shutil
from datetime import datetime
import wandb
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
#from torch.utils.tensorboard import SummaryWriter
from model_all import VideoDataSet, HNeRV, HNeRVDecoder, TransformInput
from LLM_dataset import LLMDataset
from hnerv_utils import *
from torch.utils.data import Subset
from copy import deepcopy
from dahuffman import HuffmanCodec
from torchvision.utils import save_image
import pandas as pd

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig):
    yaml_conf = str(OmegaConf.to_yaml(cfg))

    OmegaConf.set_struct(cfg, False)
    print(yaml_conf)

    # === perpare args === #
    args = cfg.model
    args.llm_path = cfg.llm_path
    args.layer_type = cfg.layer_type
    args.layer_depth = cfg.layer_depth
    args.p_size = cfg.p_size
    args.wandb = cfg.wandb
    args.outf = cfg.outf

    torch.set_printoptions(precision=4)

    # === getting the experiment id === #
    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    extra_str = 'Size{}_ENC_{}_{}_DEC_{}_{}_{}{}'.format(args.modelsize,
                                                           args.conv_type[0],
                                                           args.enc_strd_str,
                                                           args.conv_type[1],
                                                           args.dec_strd_str,
                                                           '' if args.norm == 'none' else f'_{args.norm}',
                                                           '_dist' if args.distributed else '')
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    embed_str = f'{args.embed}_Dim{args.enc_dim}'
    args.vid = args.layer_type + f'{args.layer_depth}_{args.p_size}'
    exp_id = f'{args.vid}/{embed_str}_FC{args.fc_hw}_KS{args.ks}_RED{args.reduce}_low{args.lower_width}_blk{args.num_blks}' + \
            f'_e{args.epochs}_b{args.batchSize}_{args.quant_str}_lr{args.lr}_{args.lr_type}_{args.loss}_{extra_str}{args.act}{args.block_params}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    # === for multi gpu === #
    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

def data_to_gpu(x, device):
    return x.to(device)

def train(local_rank, args):
    # === GPU setting === #
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()
        args.batchSize = int(args.batchSize / args.ngpus_per_node)

    # ==== dataset === #
    train_dataset = LLMDataset(args.llm_path, args.layer_depth, args.layer_type, args.p_size, rank=64)
    args.final_size = train_dataset.final_size
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)
    args.full_data_length = len(train_dataset)

    # === Compute the parameter number === #
    if 'pe' in args.embed or 'le' in args.embed:
        embed_param = 0
        embed_dim = int(args.embed.split('_')[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds**2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio)
        embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
        args.enc_dim = f'{int(enc_dim1)}_{embed_dim}'
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    decoder_size = args.modelsize * 1e6 - embed_param
    ch_reduce = 1. / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
    fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
    a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
    b =  embed_dim * fc_param
    c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
    args.fc_dim = int(np.roots([a,b,c - decoder_size]).max())

    # Building model
    model = HNeRV(args)

    ##### get model params and flops #####
    if local_rank in [0, None]:
        encoder_param = (sum([p.data.nelement() for p in model.encoder.parameters()]) / 1e6)
        decoder_param = (sum([p.data.nelement() for p in model.decoder.parameters()]) / 1e6)
        total_param = decoder_param + embed_param / 1e6
        args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, str(total_param)
        param_str = f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M'
        print(f'{args}\n {model}\n {param_str}', flush=True)
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'{param_str}\n')

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(args.device))
    if args.distributed and args.ngpus_per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    #elif args.ngpus_per_node > 1:
        #model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        print(args.device)
        model = model.to(args.device)
    #model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()}
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt, strict=False)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt, strict=False)
        else:
            model.load_state_dict(new_ckt, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))

    # resume from model_latest
    if not args.not_resume:
        try:
            checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
            if os.path.isfile(checkpoint_path):
                print('checkpoint_path', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            else:
                print("=> No resume checkpoint found at '{}'".format(checkpoint_path))
        except RuntimeError:
            print('best model loaded instead of latest')
            checkpoint_path = os.path.join(args.outf, 'model_best.pth')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.start_epoch < 0:
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch']
        args.start_epoch = max(args.start_epoch, 0)

    wandb.init(
        **args.wandb, config=OmegaConf.to_container(args, resolve=True)
    )
    # Training
    start = datetime.now()
    best_epoch, best_epoch_quant = 0, 0
    best_frob, best_frob_quant = 100000, 1000000
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        A_hat = {block: torch.zeros_like(train_dataset.A[block]) for block in train_dataset.A.keys()}
        epoch_loss = 0
        # iterate over dataloader
        device = args.device
        for i, sample in enumerate(train_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
            blocks = sample['layer_depth']
            # forward and backward
            cur_input = norm_idx if 'pe' in args.embed else img_data
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            #print(model.device, cur_input.device)
            img_out, _, _ = model(cur_input)
            final_loss = loss_fn(img_out, img_data, args.loss)
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            epoch_loss += final_loss.item()
            all_recon = {}
            for l in range(train_dataloader.batch_size):
                # A_hat = torch.zeros_like(train_dataset.A[blocks[l]])
                j, k = train_dataset.coord[img_idx[l]].cpu().numpy()
                cur_layer = int(blocks[l].item())
                dev = A_hat[cur_layer]
                A_hat[cur_layer][train_dataset.p_size * j:train_dataset.p_size * (j + 1), train_dataset.p_size * k:train_dataset.p_size * (k + 1)] = img_out[l, 0].detach().to(dev)
            pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_data))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr,
                    RoundTensor(pred_psnr, 2))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])
            torch.distributed.reduce(A_hat.to(local_rank), dst=0)


        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            metrics = {'psnr': pred_psnr, 'lr': lr, 'best_epoch': best_epoch}
            frob = 0
            for block in A_hat.keys():
                A_hat_b = A_hat[block].cpu() * 2 - 1
                A_hat_b = A_hat_b * train_dataset.scales_q[block]
                A_hat_b = A_hat_b + train_dataset.L1[block] @ train_dataset.L2[block]
                metrics['frob/'+str(block)] = torch.linalg.norm(A_hat_b - train_dataset.A[block])
                frob += metrics['frob/'+str(block)]
            frob /= len(A_hat.keys())
            epoch_loss /= train_dataloader.batch_size
            metrics['loss'] = epoch_loss
            metrics['frobenius'] = frob
            wandb.log(metrics, step=epoch) # TODO add the best epoch
            #writer.add_scalar(f'Train/pred_PSNR_{h}X{w}', pred_psnr, epoch+1)
            #writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            best_frob_quant, best_epoch_quant = evaluate(epoch, best_frob_quant, best_epoch_quant,
                                                         model, train_dataloader, local_rank, args,
                                                         True)  # True if epoch == args.epochs - 1 else False)

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }
        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if frob <= best_frob:
                torch.save(save_checkpoint, f'{args.outf}/model_best.pth')
                best_frob = frob
                best_epoch = epoch

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")


@torch.no_grad()
def evaluate(epoch, best_frob_quant, best_epoch_quant, model, train_dataloader, local_rank, args, huffman_coding=True):
    model_q, quant_ckt = quant_model(model, args)

    pred_psnr_list = []
    epoch_loss = 0
    # A_hat = torch.zeros_like(train_dataloader.dataset.A)
    A_hat = {block: torch.zeros_like(train_dataloader.dataset.A[block]) for block in train_dataloader.dataset.A.keys()}
    # iterate over dataloader
    device = args.device
    patch_idx_lst, all_embs = [], []
    # quant_lst, quant_min_lst, quant_scale_lst) = [], [], [], []
    for i, sample in enumerate(train_dataloader):
        img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'],
                                                                                      device), data_to_gpu(
            sample['idx'], device)
        blocks = sample['layer_depth']
        patch_idx_lst.append(img_idx.cpu())
        # forward and backward
        cur_input = norm_idx if 'pe' in args.embed else img_data
        #_, vid_embed, _ = model(cur_input)  # we get the embedding to quantize it and pass it into the model_q
        #quant_embed, dequant_emved = quant_tensor(vid_embed[0], args.quant_embed_bit)
        #quant_lst.append(quant_embed['quant'])
        #quant_min_lst.append(quant_embed['min'].unsqueeze(0))
        #quant_scale_lst.append(quant_embed['scale'].unsqueeze(0))
        img_out, vid_embed, _ = model_q(cur_input, None) # , dequant_emved)
        all_embs.append(vid_embed[0])
        final_loss = loss_fn(img_out, img_data, args.loss)
        epoch_loss += final_loss.item()

        '''
        for l in range(train_dataloader.batch_size):
            j, k = train_dataloader.dataset.coord[img_idx[l]].cpu().numpy()
            p_size = train_dataloader.dataset.p_size
            A_hat[p_size * j:p_size * (j + 1),p_size * k:p_size * (k + 1)] = img_out[l, 0].detach().to(A_hat.device)
        '''
        for l in range(train_dataloader.batch_size):
            # A_hat = torch.zeros_like(train_dataset.A[blocks[l]])
            j, k = train_dataloader.dataset.coord[img_idx[l]].cpu().numpy()
            cur_layer = int(blocks[l].item())
            dev = A_hat[cur_layer].device
            A_hat[cur_layer][train_dataloader.dataset.p_size * j:train_dataloader.dataset.p_size * (j + 1),
            train_dataloader.dataset.p_size * k:train_dataloader.dataset.p_size * (k + 1)] = img_out[l, 0].detach().to(dev)

        pred_psnr_list.append(psnr_fn_single(img_out.detach(), img_data))
        pred_psnr = torch.cat(pred_psnr_list).mean()

    # collect numbers from other gpus
    if args.distributed and args.ngpus_per_node > 1:
        pred_psnr = all_reduce([pred_psnr.to(local_rank)])
        torch.distributed.reduce(A_hat.to(local_rank), dst=0)
        #quant_lst = all_gather(quant_lst)
        #quant_min_lst = all_gather(quant_min_lst)
        #quant_scale_lst = all_gather(quant_scale_lst)
        all_embs = all_gather(all_embs)

    # ADD train_PSNR TO TENSORBOARD
    if local_rank in [0, None]:
        metrics = {'quant/psnr': pred_psnr}
        frob = 0
        for block in A_hat.keys():
            A_hat_b = A_hat[block].cpu() * 2 - 1
            A_hat_b = A_hat_b * train_dataloader.dataset.scales_q[block]
            A_hat_b = A_hat_b + train_dataloader.dataset.L1[block] @ train_dataloader.dataset.L2[block]
            metrics['quant/frob/' + str(block)] = torch.linalg.norm(A_hat_b - train_dataloader.dataset.A[block])
            frob += metrics['quant/frob/' + str(block)]
        frob /= len(A_hat.keys())
        epoch_loss /= train_dataloader.batch_size
        metrics['quant/loss'] = epoch_loss
        metrics['quant/frobenius'] = frob
        wandb.log(metrics, step=epoch)

    '''
    if local_rank in [0, None]:
        A_hat = A_hat.cpu() * 2 - 1
        A_hat = A_hat * train_dataloader.dataset.scales_q
        A_hat = A_hat + train_dataloader.dataset.L1 @ train_dataloader.dataset.L2
        frob = torch.linalg.norm(A_hat - train_dataloader.dataset.A)
        epoch_loss /= train_dataloader.batch_size
        metrics = {"quant/loss": epoch_loss, 'quant/frobenius': frob, 'quant/psnr': pred_psnr}  # TODO add the best epoch for quant
    '''
    # dump quantized checkpoint, and decoder
    if local_rank in [0, None] and quant_ckt != None:
        # vid_embed = torch.cat(quant_vid_lst)
        quant_vid = {#'quant': torch.cat(quant_lst),
                     #'min': torch.cat(quant_min_lst),
                     #'scale': torch.cat(quant_scale_lst),
                     'embs': torch.cat(all_embs),
                     'patch_idx': torch.cat(patch_idx_lst),
                     'model': quant_ckt}
        torch.save(quant_vid, f'{args.outf}/quant_vid.pth')
        if frob <= best_frob_quant:
            best_frob_quant = frob
            best_epoch_quant = epoch
            torch.save(quant_vid, f'{args.outf}/best_quant_vid.pth')
        # torch.jit.save(torch.jit.trace(HNeRVDecoder(model), (vid_embed[:2])), f'{args.outf}/img_decoder.pth')
        # huffman coding
        if huffman_coding:
            quant_v_list = []                   #  quant_vid['quant'].flatten().tolist().
            tmin_scale_len = quant_vid['embs'].flatten().nelement()
            # tmin_scale_len = quant_vid['min'].nelement() + quant_vid['scale'].nelement()
            for k, layer_wt in quant_ckt.items():
                quant_v_list.extend(layer_wt['quant'].flatten().tolist())
                tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

            # get the element name and its frequency
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]

            # total bits for quantized embed + model weights
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            args.bits_per_param = str(total_bits / len(quant_v_list))

            # including the overhead for min and scale storage,
            total_bits += tmin_scale_len * 16  # (16bits for float16)
            args.full_bits_per_param = str(total_bits / len(quant_v_list))

            # bits per pixel
            args.total_bpp = str(total_bits / float(args.final_size) / float(args.full_data_length))
            print(
                f'After quantization and encoding: \n bits per parameter:{round(float(args.full_bits_per_param), 2)}, bits per pixel: {round(float(args.total_bpp), 4)}')
    return best_frob_quant, best_epoch_quant

def quant_model(model, args):
    cur_model = deepcopy(model)
    quant_ckt, cur_ckt = [model.state_dict() for _ in range(2)]
    encoder_k_list = []
    for k,v in cur_ckt.items():
        if 'encoder' in k or 'head' in k or 'bias' in k:
            encoder_k_list.append(k)
        else:
            quant_v, new_v = quant_tensor(v, args.quant_model_bit)
            quant_ckt[k] = quant_v  # in 8bits
            cur_ckt[k] = new_v  # approximation of in args.quant_model_bit bits
    for encoder_k in encoder_k_list:
        del quant_ckt[encoder_k]
    cur_model.load_state_dict(cur_ckt)
    return cur_model, quant_ckt


if __name__ == '__main__':
    main()
