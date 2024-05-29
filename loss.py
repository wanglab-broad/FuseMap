import torch.nn.functional as F
import sklearn
import numpy as np
import torch
import torch.distributions as D
import pandas as pd
from sparse import COO

def AE_Gene_loss( recon_x, x, z_distribution):
    if recon_x.shape[0]==0:
        return torch.tensor(0.0 ,dtype=torch.float32).to(recon_x.device)

    reconstruction_loss = F.mse_loss(recon_x ,x)
    kl_divergence = D.kl_divergence(z_distribution, D.Normal(0.0, 1.0)).sum(dim=1).mean() / x.shape[1]
    return reconstruction_loss + kl_divergence


def prod(x):
    ########### function from GLUE: https://github.com/gao-lab/GLUE
    # try:
        # from math import prod  # pylint: disable=redefined-outer-name
    #     return np.prod(x)
    # except ImportError:
        ans = 1
        for item in x:
            ans = ans * item
        return ans

'''
pretrain loss
'''
def compute_dis_loss_pretrain(model, flag_source_cat_single, flag_source_cat_spatial, anneal,
                              batch_features_all, adj_all, mask_batch_single, mask_batch_spatial,args):

    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)


    z_all = [model.encoder['atlas' +str(i)]( batch_features_all[i], adj_all[i] )
             for i in range(args.n_atlas)]
    z_mean_cat_single = torch.cat([z_all[i][0] for i in range(args.n_atlas)])[mask_batch_single_all ,:]

    z_spatial_all = [z_all[i][2] for i in range(args.n_atlas)]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all ,:]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample((z_mean_cat_single.shape[0], ))
            z_mean_cat_single = z_mean_cat_single + (anneal * args.align_noise_coef) * noise_single
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(0, args.EPS+ z_mean_cat_spatial.std(axis=0)).sample((z_mean_cat_spatial.shape[0], ))
            z_mean_cat_spatial = z_mean_cat_spatial + (anneal * args.align_noise_coef) * noise_spatial

    ### compute dis loss
    loss_dis_single = F.cross_entropy(F.softmax(model.discriminator_single(z_mean_cat_single) ,dim=1),
                                      flag_source_cat_single[mask_batch_single_all] ,reduction="none")
    loss_dis_single = loss_dis_single.sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(F.softmax(model.discriminator_spatial(z_mean_cat_spatial) ,dim=1),
                                       flag_source_cat_spatial[mask_batch_spatial_all] ,reduction="none")
    loss_dis_spatial = loss_dis_spatial.sum() / loss_dis_spatial.numel()


    loss_dis = args.lambda_disc_single * (loss_dis_single + loss_dis_spatial)
    # loss_dis = lambda_disc_single * (loss_dis_single )

    loss_all = { "dis": loss_dis}
    return loss_all



def compute_ae_loss_pretrain(model, flag_source_cat_single, flag_source_cat_spatial,
                             anneal, batch_features_all, adj_all,
                             mask_batch_single, mask_batch_spatial, args):

    z_all = [model.encoder['atlas' +str(i)]( batch_features_all[i],  adj_all[i] )
             for i in range(args.n_atlas)]

    z_distribution_all = [D.Normal(z_all[i][0], z_all[i][1]) for i in range(args.n_atlas)]


    z_sample_all = [z_distribution_all[i].rsample() for i in range(args.n_atlas)]

    z_spatial_all = [z_all[i][2] for i in range(args.n_atlas)]


    decoder_all = [model.decoder['atlas' +str(i)]( z_sample_all[i] ,z_spatial_all[i], adj_all[i]) for i in range(args.n_atlas)]

    ### compute AE loss
    z_distribution_loss = [D.Normal(z_all[i][0][mask_batch_single[i] ,:], z_all[i][1][mask_batch_single[i] ,:]) for i in range(args.n_atlas)]
    loss_AE_all = [args.lambda_ae_single * AE_Gene_loss(decoder_all[i][ mask_batch_single[i] ,: ],
                                                             batch_features_all[i][ mask_batch_single[i] ,: ],
                                                             z_distribution_loss[i],
                                                             ) for i in range(args.n_atlas)]

    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)

    z_mean_cat_single = torch.cat([z_all[i][0] for i in range(args.n_atlas)])[mask_batch_single_all ,:]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all ,:]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample((z_mean_cat_single.shape[0], ))
            z_mean_cat_single = z_mean_cat_single + (anneal * args.align_noise_coef) * noise_single
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(0, args.EPS+ z_mean_cat_spatial.std(axis=0)).sample((z_mean_cat_spatial.shape[0], ))
            z_mean_cat_spatial = z_mean_cat_spatial + (anneal * args.align_noise_coef) * noise_spatial


    ### compute dis loss

    loss_dis_single = F.cross_entropy(F.softmax(model.discriminator_single(z_mean_cat_single), dim=1),
                                      flag_source_cat_single[mask_batch_single_all],
                                      reduction="none")
    loss_dis_single =  loss_dis_single.sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(F.softmax(model.discriminator_spatial(z_mean_cat_spatial), dim=1),
                                       flag_source_cat_spatial[mask_batch_spatial_all],
                                       reduction="none")
    loss_dis_spatial = loss_dis_spatial.sum() / loss_dis_spatial.numel()

    loss_dis = args.lambda_disc_single * (loss_dis_single + loss_dis_spatial)

    if args.lambda_disc_single==1  :# and loss_dis.item()<sum(loss_AE_all).item()/DIS_LAMDA:
        args.lambda_disc_single =sum(loss_AE_all).item( ) / args.DIS_LAMDA /loss_dis.item()
        print(f'lambda_disc_single changed to {args.lambda_disc_single}')
        loss_dis = args.lambda_disc_single * loss_dis

    loss_all = {
        "dis_ae": loss_dis,
        'loss_AE_all': loss_AE_all,
        'loss_all': - loss_dis + sum(loss_AE_all), }
    return loss_all


'''
final train loss
'''
def compute_dis_loss(model, flag_source_cat_single, flag_source_cat_spatial, anneal,
                     batch_features_all, adj_all, mask_batch_single, mask_batch_spatial,
                     balance_weight_single_block ,balance_weight_spatial_block, args):

    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)
    balance_weight_single_block = torch.hstack(balance_weight_single_block)
    balance_weight_spatial_block = torch.hstack((balance_weight_spatial_block))

    z_all = [model.encoder['atlas' +str(i)]( batch_features_all[i], adj_all[i] )
             for i in range(args.n_atlas)]
    z_mean_cat_single = torch.cat([z_all[i][0] for i in range(args.n_atlas)])[mask_batch_single_all ,:]

    z_spatial_all = [z_all[i][2] for i in range(args.n_atlas)]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all ,:]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample((z_mean_cat_single.shape[0], ))
            z_mean_cat_single = z_mean_cat_single + (anneal * args.align_noise_coef) * noise_single
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(0, args.EPS+ z_mean_cat_spatial.std(axis=0)).sample((z_mean_cat_spatial.shape[0], ))
            z_mean_cat_spatial = z_mean_cat_spatial + (anneal * args.align_noise_coef) * noise_spatial

    ### compute dis loss
    loss_dis_single = F.cross_entropy(F.softmax(model.discriminator_single(z_mean_cat_single) ,dim=1),
                                      flag_source_cat_single[mask_batch_single_all] ,reduction="none")
    loss_dis_single = (balance_weight_single_block
                           [mask_batch_single_all] * loss_dis_single).sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(F.softmax(model.discriminator_spatial(z_mean_cat_spatial) ,dim=1),
                                       flag_source_cat_spatial[mask_batch_spatial_all] ,reduction="none")
    loss_dis_spatial = (balance_weight_spatial_block
                            [mask_batch_spatial_all] * loss_dis_spatial).sum() / loss_dis_spatial.numel()


    loss_dis = args.lambda_disc_single * (loss_dis_single + loss_dis_spatial)

    loss_all = { "dis": loss_dis}
    return loss_all


def compute_ae_loss(model, flag_source_cat_single, flag_source_cat_spatial,
                    anneal, batch_features_all, adj_all,
                    mask_batch_single, mask_batch_spatial,
                    balance_weight_single_block, balance_weight_spatial_block, args):

    z_all = [model.encoder['atlas' +str(i)]( batch_features_all[i],  adj_all[i] )
             for i in range(args.n_atlas)]

    z_distribution_all = [D.Normal(z_all[i][0], z_all[i][1]) for i in range(args.n_atlas)]
    z_sample_all = [z_distribution_all[i].rsample() for i in range(args.n_atlas)]

    z_spatial_all = [z_all[i][2] for i in range(args.n_atlas)]


    decoder_all = [model.decoder['atlas' +str(i)]( z_sample_all[i] ,z_spatial_all[i], adj_all[i]) for i in range(args.n_atlas)]

    ### compute AE loss
    z_distribution_loss = [D.Normal(z_all[i][0][mask_batch_single[i] ,:], z_all[i][1][mask_batch_single[i] ,:]) for i in range(args.n_atlas)]
    loss_AE_all = [args.lambda_ae_single * AE_Gene_loss(decoder_all[i][ mask_batch_single[i] ,: ],
                                                             batch_features_all[i][ mask_batch_single[i] ,: ],
                                                             z_distribution_loss[i],
                                                             ) for i in range(args.n_atlas)]

    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)

    z_mean_cat_single = torch.cat([z_all[i][0] for i in range(args.n_atlas)])[mask_batch_single_all ,:]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all ,:]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample((z_mean_cat_single.shape[0], ))
            z_mean_cat_single = z_mean_cat_single + (anneal * args.align_noise_coef) * noise_single
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(0, args.EPS+ z_mean_cat_spatial.std(axis=0)).sample((z_mean_cat_spatial.shape[0], ))
            z_mean_cat_spatial = z_mean_cat_spatial + (anneal * args.align_noise_coef) * noise_spatial


    ### compute dis loss
    balance_weight_single_block = torch.hstack(balance_weight_single_block)
    balance_weight_spatial_block = torch.hstack((balance_weight_spatial_block))

    loss_dis_single = F.cross_entropy(F.softmax(model.discriminator_single(z_mean_cat_single), dim=1),
                                      flag_source_cat_single[mask_batch_single_all],
                                      reduction="none")
    loss_dis_single = (balance_weight_single_block
                           [mask_batch_single_all] * loss_dis_single).sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(F.softmax(model.discriminator_spatial(z_mean_cat_spatial), dim=1),
                                       flag_source_cat_spatial[mask_batch_spatial_all],
                                       reduction="none")
    loss_dis_spatial = (balance_weight_spatial_block
                            [mask_batch_spatial_all] * loss_dis_spatial).sum() / loss_dis_spatial.numel()


    loss_dis = args.lambda_disc_single * (loss_dis_single + loss_dis_spatial)


    if args.lambda_disc_single==1  :# and loss_dis.item()<sum(loss_AE_all).item()/DIS_LAMDA:
        lambda_disc_single =sum(loss_AE_all).item( ) / args.DIS_LAMDA /loss_dis.item()
        print(f'lambda_disc_single changed to {lambda_disc_single}')
        loss_dis = lambda_disc_single * loss_dis

    loss_all = {
        "dis_ae": loss_dis,
        'loss_AE_all': loss_AE_all,
        'loss_all': - loss_dis + sum(loss_AE_all), }
    return loss_all




'''
balance weight part
'''


def get_balance_weight_subsample( leiden_adata_single, adatas_ ,key_leiden_category):
    ########### function from GLUE: https://github.com/gao-lab/GLUE
    us = [sklearn.preprocessing.normalize(leiden.X, norm="l2") for leiden in leiden_adata_single]
    ns = [leiden.obs["size"] for leiden in leiden_adata_single]

    power = 4
    cutoff = 0.5
    while True:
        summary_balance_dict_sum = {}
        summary_balance_dict_multiply = {}
        summary_balance_dict_num ={}
        for i, ui in enumerate(us):
            for j, uj in enumerate(us[i + 1:], start=i + 1):
                cosine = ui @ uj.T
                cosine[cosine < cutoff] = 0
                cosine = COO.from_numpy(cosine)
                cosine = np.power(cosine, power)

                for ind in [i ,j]:
                    if ind==i:
                        balancing = cosine.sum(axis=1).todense() / ns[ind]
                    else:
                        balancing = cosine.sum(axis=0).todense() / ns[ind]
                    balancing = pd.Series(balancing, index=leiden_adata_single[ind].obs_names)
                    balancing = balancing.loc[adatas_[ind].obs[key_leiden_category]].to_numpy()
                    balancing /= balancing.sum() / balancing.size
                    if ind in summary_balance_dict_sum:
                        summary_balance_dict_sum[ind] += balancing.copy()
                        summary_balance_dict_multiply[ind] *= balancing.copy()
                        summary_balance_dict_num[ind] += 1
                    else:
                        summary_balance_dict_sum[ind] = balancing.copy()
                        summary_balance_dict_multiply[ind] = balancing.copy()
                        summary_balance_dict_num[ind ] =1
        flag =0
        for i in range(len(summary_balance_dict_sum)):
            if sum(np.isnan(summary_balance_dict_sum[i]) ) >0:
                flag =1
                break
        for i in range(len(summary_balance_dict_multiply)):
            if sum(np.isnan(summary_balance_dict_multiply[i])) > 0:
                flag = 1
                break
        for i in range(len(summary_balance_dict_multiply)):
            if sum(summary_balance_dict_sum[i]) == 0:
                flag = 1
                break
        for i in range(len(summary_balance_dict_multiply)):
            if sum(summary_balance_dict_multiply[i] )==0:
                flag = 1
                break
        if flag==1:
            cutoff-=0.1
        else:
            break
    print(f'balance weight final cutoff: {cutoff}')
    for i in range(len(summary_balance_dict_sum)):
        if summary_balance_dict_sum[i][summary_balance_dict_sum[i] == np.inf].shape[0] > 0:
            print(i, 'inf:', summary_balance_dict_sum[i][summary_balance_dict_sum[i] == np.inf].shape[0])
            summary_balance_dict_sum[i][summary_balance_dict_sum[i] == np.inf] = 1e308

    for i in range(len(summary_balance_dict_sum)):
        if summary_balance_dict_multiply[i][summary_balance_dict_multiply[i] == np.inf].shape[0] > 0:
            print(i, 'inf:', summary_balance_dict_multiply[i][summary_balance_dict_multiply[i] == np.inf].shape[0])
            summary_balance_dict_multiply[i][summary_balance_dict_multiply[i] == np.inf] = 1e308

    balance_weight = []
    summary_balance_dict ={}
    for i in range(len(us)):
        test1 = summary_balance_dict_sum[i ] /(summary_balance_dict_sum[i].sum() / summary_balance_dict_sum[i].size)
        test2 =  summary_balance_dict_multiply[i ] /( summary_balance_dict_multiply[i].sum() / summary_balance_dict_multiply[i].size)
        test = 0.9* test1 + 0.1 * test2
        test /= test.sum() / test.size
        summary_balance_dict[i ] =test.copy()
        balance_weight.append(summary_balance_dict[i] )
    return balance_weight

def get_balance_weight(adatas, leiden_adata_single, adatas_ ,key_leiden_category):
    ########### function from GLUE: https://github.com/gao-lab/GLUE
    us = [sklearn.preprocessing.normalize(leiden.X, norm="l2") for leiden in leiden_adata_single]
    ns = [leiden.obs["size"] for leiden in leiden_adata_single]

    cosines = []
    cutoff = 0.5
    power = 4

    for i, ui in enumerate(us):
        for j, uj in enumerate(us[i + 1:], start=i + 1):
            cosine = ui @ uj.T
            cosine[cosine < cutoff] = 0
            cosine = COO.from_numpy(cosine)
            cosine = np.power(cosine, power)
            key = tuple(
                slice(None) if k in (i, j) else np.newaxis
                for k in range(len(us))
            )  # To align axes
            cosines.append(cosine[key])
    joint_cosine = prod(cosines)

    if joint_cosine.coords.shape[0] == 0:
        raise ValueError('Balance weight computation error! No correlation between samples or lower cutoff!')
    #
    balance_weight = []
    for i, (adata, adata_, leiden, n) in enumerate(zip(adatas, adatas_, leiden_adata_single, ns)):
        balancing = joint_cosine.sum(axis=tuple(
            k for k in range(joint_cosine.ndim) if k != i
        )).todense() / n
        balancing = pd.Series(balancing, index=leiden.obs_names)
        balancing = balancing.loc[adata_.obs[key_leiden_category]].to_numpy()
        balancing /= balancing.sum() / balancing.size
        balance_weight.append(balancing)
    return balance_weight

'''
train ref data part
'''

def _load_snapshot(snapshot_path, loc):
    snapshot = torch.load(snapshot_path, map_location=loc)
    model.load_state_dict(snapshot["MODEL_STATE"])
    epochs_run_pretrain = snapshot["EPOCHS_RUN_pretrain"]
    epochs_run_final = snapshot["EPOCHS_RUN_final"]
    print \
        (f" Resuming training from snapshot at pretrain Epoch {epochs_run_pretrain}, final epoch {epochs_run_final}\n")

def _save_snapshot(epoch_pretrain, epoch_final):
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "EPOCHS_RUN_pretrain": epoch_pretrain,
        "EPOCHS_RUN_final" :epoch_final,
    }
    torch.save(snapshot, snapshot_path)
    print \
        (f"Pretrain Epoch {epoch_pretrain}, final Epoch {epoch_final} | Training snapshot saved at {snapshot_path}\n")


