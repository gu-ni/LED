import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.text_datasets import load_data_text

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_taae_vocab
)

from TAAE.model_TAE import AAE
from TAAE.vocab import Vocab


def create_argparser():
    defaults = dict(model_path='trained_diffuseq/ema_0.9999_014000.pt', step=500, out_dir='', top_p=0, batch_size=50)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(os.path.join(args.model_path))
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_taae_vocab(args)
    vocab_file = 'TAAE/vocab.txt'
    vocab = Vocab(vocab_file)
    ckpt = th.load(os.path.join(args.tae_model_vocab_path, 'model.pt'), map_location='cpu')
    tae_model = AAE(vocab, ckpt['args']).to(dist_util.dev()) # set device
    tae_model.load_state_dict(ckpt['model'])
    tae_model.eval()
    
    for param in tae_model.parameters():
        param.requires_grad = False
 
    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args = args,
        split=args.split,
        loaded_vocab=tokenizer,
        loop=False
    )
    
    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    idx = 0

    try:
        while True:
            cond = next(data_valid)
            # print(batch.shape)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print('### End of reading iteration...')

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    for cond in iterator:

        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier()
            continue
        
        in_x = cond.pop('input_id_x').to(dist_util.dev())
        in_y = cond.pop('input_id_y').to(dist_util.dev())
        mu_x, _, _ = tae_model.encode(in_x.permute(1, 0))
        mu_y, _, _ = tae_model.encode(in_y.permute(1, 0))
        z_x = mu_x
        z_y = mu_y
        x_start = th.cat([z_x, z_y], dim=0).permute(1, 0, 2).contiguous()
        
        # x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')

        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=step_gap
        )
        # print(samples[0].shape) # samples for each step

        sample = samples[-1]

        _, y_temp = th.split(sample.permute(1, 0, 2).contiguous(), 64, dim=0)

        pred_sentence = th.ones(y_temp.shape[0], y_temp.shape[1]).long().to(dist_util.dev())*vocab.go
        memory = tae_model.drop(tae_model.z2emb(y_temp))
        for t in range(1, y_temp.shape[0]):
            _input = tae_model.position_encoding(tae_model.embed(pred_sentence))
            _input = _input[:t,:]
            tgt_mask = th.nn.Transformer.generate_square_subsequent_mask(t).to(dist_util.dev())
            
            output = tae_model.Decoder(_input, memory, tgt_mask)
            output = tae_model.drop(output)
            logits = tae_model.proj(output.view(-1, output.size(-1)))
            pred_next = logits.view(output.size(0), output.size(1), -1).argmax(dim=-1)
            pred_sentence[1:t+1, ] = pred_next

        word_lst_recover = []
        word_lst_source = []
        word_lst_ref = []
        all_word = [word_lst_recover, word_lst_source, word_lst_ref]
        
        pred_sentence = pred_sentence.t()
        for temp in zip(pred_sentence, in_x, in_y):
            for i, (sen, stop) in enumerate(zip(temp, [vocab.eos, vocab.pad, vocab.pad])):
                try:
                    index = (sen == stop).nonzero(as_tuple=False)[0, 0]
                    sen = th.split(sen, index)[0][1:-1]
                except:
                    pass
                result = ' '.join(map(lambda x: vocab.idx2word[x], sen))
                all_word[i].append(result)
                

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, 'a')
                for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                    print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
                fout.close()
            dist.barrier()

        print('abc')
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
