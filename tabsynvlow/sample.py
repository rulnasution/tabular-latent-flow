import torch
import argparse
import warnings
import time
import os
import delu
import torchsde

from tqdm import tqdm
from .latent_utils import get_input_generate, recover_data, split_num_cat_target
from .flow_matching import *
from .networks import *

warnings.filterwarnings('ignore')

def main(args):
    delu.random.seed(args.seed) ## default 42
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = args.device
    save_path = args.save_path

    model_save_path = f'{curr_dir}/ckpt/{dataname}'

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]
    print(train_z.shape)

    mean = train_z.mean(0)

    def sampling_data_ode(n_samples, t = 1.0, batch_size = 10000):
        # Sampling
        print(f'ODE integration until t={t} using {args.int_method}')
        n_times = (n_samples // batch_size) + 1
        epoch_iterator = tqdm(range(n_times))
        x_1_hat = []
        for epoch in range(n_times):
            with torch.no_grad():
                # x_0 = torch.cat([torch.randn(n_samples, d_cont, device=device),
                #                 sample_cat(n_samples, K)], dim=1)
                x_0 = torch.randn(batch_size, in_dim, device=device)
                # x_1_hat.append(v_t.decode(x_0))
                x_1_hat.append(v_t.decode_t0_t1(x_0, 0.0, t, method = args.int_method))

        x_1_hat = torch.cat(x_1_hat, dim = 0)
        return x_1_hat[:n_samples,:]

    def sampling_data_sde(n_samples, t = 1.0, batch_size = 10000):
        '''
        "euler"
        "milstein"
        "heun" (for Stratonovich)
        "midpoint" (for Stratonovich)
        '''
        print(f'SDE integration until t={t} using {args.steps} steps using {args.int_method}')
        # Sampling
        v_t.eval()
        n_times = (n_samples // batch_size) + 1
        ts = torch.linspace(0, t, 2).to('cuda:0')
        # epoch_iterator = tqdm(range(n_times))
        x_1_hat = []
        if args.int_method in ['heun', 'midpoint']: v_t.sde_type = "stratonovich"
        for epoch in range(n_times):
            with torch.no_grad():
                x_0 = torch.randn(batch_size, in_dim, device=device)
                res1 = torchsde.sdeint(v_t, x_0, ts, method=args.int_method, dt = 1/args.steps)
                x_1_hat.append(res1[-1].detach())
            # x_1_hat.append(v_t.decode(x_0).detach())
        v_t.train()

        x_1_hat = torch.cat(x_1_hat, dim = 0)
        return x_1_hat[:n_samples,:]
        
    print('Start sampling...')
    start_time = time.time()
    
    learnable = args.var_weight == 'learnable'
    var_caption = '' if args.var_weight == 'efvfm' else f'_{args.var_weight}' ## default is the empirical efvfm in tabsynvlow
    net = Net(in_dim, 512, learnable=learnable).to(device) ## already according to tabsyn paper
    if args.saved_epoch == 0: ## don't forget to add _{args.cond_vel}
        print(f'tabsynvlow uses {args.cond_vel} trajectory and best model from {args.var_weight} loss')
        net.load_state_dict(torch.load(f'{model_save_path}/model{var_caption}_{args.cond_vel}.pt'))
    else:
        print(f'tabsynvlow uses {args.cond_vel} trajectory and {args.saved_epoch} epoch model from {args.var_weight} loss')
        net.load_state_dict(torch.load(f'{model_save_path}/per_epoch/model{var_caption}_{args.cond_vel}_{args.saved_epoch}.pt'))
    
    num_samples = train_z.shape[0]

    match args.cond_vel:
        case 'ot': net_t = OT_t().to(device)
        case 'vp': net_t = VPDiffusion_t().to(device)
        case 've': net_t = VEDiffusion_t().to(device)
        case 'logit': net_t = LogitNormal_t().to(device)
        case 'cos': net_t = Cosine_t().to(device)
        case _: raise Exception(f'Unknown conditional velocity formula: {args.cond_vel}, should be between "ot", "vp" and "ve", "logit", and "cos"')

    if not args.sde:
        print('Sampling using ODE')
        v_t = CondVF(net, net_t, n_steps = args.steps).to(device)
        v_t.learnable = learnable
        x_next = sampling_data_ode(num_samples, t = args.t_ode, batch_size= num_samples if args.batch_size == 0 else args.batch_size)
    else:
        match args.cond_vel_sigma:
            case 'ot': net_t_sigma = OT_t().to(device)
            case 'vp': net_t_sigma = VPDiffusion_t().to(device)
            case 've': net_t_sigma = VEDiffusion_t().to(device)
            case 'logit': net_t_sigma = LogitNormal_t().to(device)
            case 'cos': net_t_sigma = Cosine_t().to(device)
            case _: raise Exception(f'Unknown sigma formula: {args.cond_vel}, should be between "ot", "vp" and "ve", "logit", and "cos"')

        print('Sampling using SDE')
        v_t = StochasticCondVF(net, net_t, net_t_sigma, sigma_max = args.sigma_max)
        v_t.learnable = learnable
        
        '''since torchsde cannot overcome the Inf problem when we divide by beta, 
           we need to clip the t manually when t -> 1'''
        if args.cond_vel in ['vp', 'cos'] and args.t_ode == 1.:
            t_target = args.t_ode - 1e-5
        else:
            t_target = args.t_ode
        x_next = sampling_data_sde(num_samples, t = t_target, batch_size= num_samples if args.batch_size == 0 else args.batch_size)

    print(f'finished in {v_t.nfe} steps')
    v_t.nfe = 0
    
    '''
        Generating samples    
    '''
    start_time = time.time()

    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'