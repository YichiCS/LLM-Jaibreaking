import torch
import numpy as np
import pandas as pd

from utils import *
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from loss import batch_log_bleulosscnn_ae

stop_words = set(stopwords.words('english'))

def decode(model, tokenizer, device, x="", z="", args=None, sys_prompt=None, q_id=None):
    
    model.eval()
    
    bad_words = get_badwords()

    x_ = tokenizer.encode(x)[1:]
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(args.batch_size, 1)
    x_onehot = x_onehot.repeat(args.batch_size, 1, 1)

    z_mask = None
    x_mask = None
    # extract keywords:
    z_ = tokenizer.encode(z)[1:]  
    z_t = torch.tensor(z_, device=device, dtype=torch.long)

    z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
    z_onehot = z_onehot.repeat(args.batch_size, 1, 1)

    z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)

    length = args.length
    if length <= 0:
        length = z_t.shape[1] - length
    
    print(f"[x]: |{tokenizer.decode(x_)}|")
    print(f"[z]: |{tokenizer.decode(z_)}|")
    print(f"[length]: {length}")

    # z_mask: [batch_size, vocab_size]
    z_words = word_tokenize(z[:])  
    z_nonstop_words = [w.lower() for w in z_words if w.lower() not in stop_words and w.isalnum()]
    z_nonstop_words += [z_words[0]]
    z_nonstop_words = ' ' + ' '.join(z_nonstop_words)
    z_nonstop_ = tokenizer.encode(z_nonstop_words)
    

    z_mask = np.zeros([tokenizer.vocab_size])
    z_mask[z_nonstop_] = 1.
    z_mask = torch.tensor(z_mask, device=device)
    z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)
    
    length = args.length

    x_words = tokenizer.encode(bad_words)
    x_mask = np.zeros([tokenizer.vocab_size])
    x_mask[x_words] = 1.
    x_mask = torch.tensor(x_mask, device=device)

    bad_mask = x_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)
    bad_mask = torch.ones_like(bad_mask, device=device) - bad_mask
    bad_words_ = tokenizer.encode(bad_words)[:]
    bad_words_t = torch.tensor(bad_words_, device=device, dtype=torch.long)

    bad_words_onehot = one_hot(bad_words_t, dimension=tokenizer.vocab_size)
    bad_words_onehot = bad_words_onehot.repeat(args.batch_size, 1, 1)

    bad_words_t = bad_words_t.unsqueeze(0).repeat(args.batch_size, 1)

    if args.init_mode == 'original':
        init_logits = initialize(model, x_t, length, args.init_temp)
    else:
        init_logits = z_onehot / 0.01
        init_logits = init_logits[:, :length, :]
        if length > init_logits.shape[1]:
            init_logits = torch.cat(
                [init_logits,
                 torch.zeros([args.batch_size, length - init_logits.shape[1], tokenizer.vocab_size], device=device)],
                dim=1)

    y_logits = init_logits

    epsilon = torch.nn.Parameter(torch.zeros_like(y_logits))
    optim = torch.optim.Adam([epsilon], lr=args.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim, 
        step_size=args.stepsize_iters,
        gamma=args.stepsize_ratio
    )

    frozen_len = args.frozen_length

    y_logits_ = None

    noise_std = 0.0
    
    soft_forward_x = x_onehot[:, -1:, :] 
    if x_t.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_t[:, :-1], use_cache=True)
        x_model_past = x_model_outputs.past_key_values

    mask_t = None
    y_logits_aug = []
    loss_aug = []
    
    # TODO: Augment Langevin

    for iter in tqdm(range(args.num_iters), ncols=100):
        optim.zero_grad()

        y_logits_ = y_logits + epsilon
        soft_forward_y = y_logits_ / 0.001
        if mask_t is None:
            soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        else:
            soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=x_mask, bad_mask=None) / 0.001
            
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)

        if args.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, args.topk)
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)
        flu_loss = soft_nll(
            top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=x_mask, bad_mask=None),
            y_logits_ / args.input_lgt_temp)

        soft_forward_y_ = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
    
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)

        # Reshaping
        bz = args.batch_size
        lg = xyz_logits.shape[1]
        st = xy_length - 1
        ed = xyz_logits.shape[1] - 1
        xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
        z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

        c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
            z_logits,
            z_t.view(-1))
        c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)

        c_loss_2 = batch_log_bleulosscnn_ae(
            decoder_outputs=y_logits_.transpose(0, 1),
            target_idx=bad_words_t,
            ngram_list=[1]
        )

        loss = args.goal_weight * c_loss_1 + 1 * flu_loss - args.rej_weight *  c_loss_2
        loss_ = loss
        loss = loss.mean()
        if iter < args.num_iters - 1: 
            loss.backward()
            optim.step()
            scheduler.step() 
            last_lr = scheduler.get_last_lr()[0]
            
            
        # Decode & Print
        if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or iter + 1 == args.num_iters):
            
            text, _, last_text_ids = decode_with_model_topk(
                model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=None, bad_mask=None)
            text_post = text
            
            for bi in range(args.batch_size):
                prompt = x + " " + text_post[bi]
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                print(bi)
                output_ids = model.generate(inputs=input_ids, temperature=0.7, max_length=512, do_sample=True, top_k=args.topk)
                
                print(f"[Model Output]: \n{tokenizer.decode(output_ids[0], skip_special_tokens=True)}")
                print(f"[Suffix {bi+1} @ {iter+1}]: {text_post[bi]}")
                print(f"Loss: {loss_[bi].item():.4f} | Loss_flu: {flu_loss[bi].item():.4f} | Loss_c1: {c_loss_1[bi].item():.4f} | Loss_c2: {c_loss_2[bi].item():.4f} | Lr: {last_lr:.4f}")
                
        ## noise
        if iter < args.num_iters - 1:

            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.
            if iter % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = args.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                noise = torch.normal(mean=args.gs_mean, std=noise_std, size=epsilon.size(),
                                     device='cuda', requires_grad=False)
                if args.win_anneal_iters >= 0 and iter >= args.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise
            
        y_logits_aug.append(y_logits_.cpu().detach())
        loss_aug.append(loss_.cpu().detach())
        
    
    y_logits_aug_t = torch.cat(y_logits_aug, dim=0)
    
    text_aug = []
    iter_N = y_logits_aug_t.shape[0] // args.decode_N + 1

    for i in tqdm(range(iter_N), ncols=100):
        
        if i*args.decode_N >= y_logits_aug_t.shape[0]:
            continue
        
        y_logits_aug_t_ = y_logits_aug_t[i*args.decode_N:min((i+1)*args.decode_N, y_logits_aug_t.shape[0])].to(device)
        
        x_onehot_decode = x_onehot[0].unsqueeze(0).repeat(y_logits_aug_t_.shape[0], 1, 1)
        x_onehot_decode = x_onehot_decode[:, -1:, :]
        x_past_decode = tuple_ex(x_model_past, y_logits_aug_t_.shape[0])
        
        text, _, _ = decode_with_model_topk(
        model, y_logits_aug_t_, args.topk, x_onehot_decode, x_past_decode, tokenizer, extra_mask=None, bad_mask=None)
        
        text_aug.append(text)
        
    torch.cuda.empty_cache()
        
    text_decode = [t for text_ in text_aug for t in text_]
    csv_path = os.path.join(args.result_path, f'prompt_{q_id}.csv')
    
    idx = [str(i) for i in range(args.batch_size)]
    data = [[] for _ in range(args.batch_size)]
    
    loss_aug = torch.cat(loss_aug, dim=0).tolist()
    
    for i in range(len(text_decode)):
        j = i%args.batch_size
        data[j].append(text_decode[i])
    data = csv_replace(data)
        
    df_dic = {idx[i]: data[i] for i in range(args.batch_size)}
    
    df = pd.DataFrame(df_dic)
    df = df.reset_index(drop=False)
    df.to_csv(csv_path, index=False)
    
    return True

