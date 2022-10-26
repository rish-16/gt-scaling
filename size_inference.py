import os, logging, datetime, shutil, time
import os.path as osp
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
from tqdm import tqdm
import torch

from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, set_agg_dir, set_cfg, load_cfg, makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric import seed_everything

from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from graphgps.encoder.kernel_pos_encoder import RWSENodeEncoder
from torch_geometric.graphgym.models.encoder import AtomEncoder
from graphgps.transform.posenc_stats import compute_posenc_stats

from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from graphgps.logger import create_logger

import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pprint import pprint

class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='../datasets/', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        """
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices

    return run_ids, seeds, split_indices

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)

def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)    

def PREPROCESS_BATCH(batch, emb_dim, dim_emb1, dim_emb2, cfg):
    """
    add lappe + rwse + atomencoder
    """
    cfg.posenc_RWSE.kernel.times = list(eval(cfg.posenc_RWSE.kernel.times_func))
    batch = compute_posenc_stats(batch, ["LapPE", "RWSE"], True, cfg)
    return batch

# if __name__ == '__main__':
#     dataset = PygPCQM4Mv2Dataset()
#     dataset = dataset[:50000]
#     # print(dataset)
#     # print(dataset.data.edge_index)
#     # print(dataset.data.edge_index.shape)
#     # print(dataset.data.x.shape)
#     # print(dataset[100])
#     # print(dataset[100].y)
#     # print(dataset.get_idx_split())

#     B = 256
#     container = {}

#     for i in range(len(dataset)):
#         g = dataset[i]
#         N = g.x.size(0)
#         if N <= 31 and N >= 1:
#             if N in container.keys():
#                 container[N].append(g)
#             else:
#                 container[N] = [g]

#     batches = []
#     # truncate to only single batch size
#     for sb, graphs in container.items():
#         container[sb] = graphs[:B]
#         print ("N:", sb, "length:", len(container[sb]))
#         batch = pyg.data.Batch.from_data_list(container[sb])
#         dl = pyg.loader.DataLoader(batch, batch_size=B)
#         batches.append(dl)

#     pprint (batches)

#     args = parse_args()

#     set_cfg(cfg)
#     load_cfg(cfg, args)
#     custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
#     dump_cfg(cfg)
    
#     # Set Pytorch environment
#     torch.set_num_threads(cfg.num_threads)
#     # gpu_dev = str(input("Enter GPU device: "))

#     # Repeat for multiple experiment runs
#     for run_id, seed, split_index in zip(*run_loop_settings()):
#         # Set configurations for each run
#         custom_set_run_dir(cfg, run_id)
#         set_printing()
#         cfg.dataset.split_index = split_index
#         cfg.seed = seed
#         cfg.run_id = run_id
#         seed_everything(cfg.seed)
        
#         DEVICE = f'cuda:0'
#         cfg.device = DEVICE

#         model = create_model()

#         # print (model)

#         cfg.params = params_count(model)
#         logging.info('Num parameters: %s', cfg.params)

        # # iterature through size classes
        # size_times = {}
        # for bi in range(len(batches)):
        #     start_time = time.time()
        #     for i, cur_batch in enumerate(batches[bi]):
        #         print (cur_batch)
        #         cur_batch = PREPROCESS_BATCH(cur_batch, 384, 8, 20, cfg)
        #         cur_batch.split = "train"

        #         cur_batch = cur_batch.to(DEVICE)
        #         print ("Current batch: ", cur_batch)
        #         print ("Current batch size: ", len(cur_batch))
        #         y1 = model(cur_batch)
        #         end_time = time.time()
        #         time_taken = end_time - start_time

        #         cur_N = cur_batch[0].x.size(0)
        #         print ("Current bucket size: ", cur_N, len(cur_batch))
        #         size_times[cur_N] = time_taken

        # pprint (size_times)

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # gpu_dev = str(input("Enter GPU device: "))
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        
        cfg.device = f'cuda:0'

        loaders = create_loader()
        train_loader = loaders[0]

        model = create_model()

        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)

        per_size_batches = {}
        for i, batch in enumerate(train_loader):
            data_batch = batch[0]
            print (data_batch)
            print (data_batch.x.size())
            break

        # # iterature through size classes
        # size_times = {}
        # for bi in range(len(batches)):
        #     start_time = time.time()
        #     for i, cur_batch in enumerate(batches[bi]):
        #         print (cur_batch)
        #         cur_batch = PREPROCESS_BATCH(cur_batch, 384, 8, 20, cfg)
        #         cur_batch.split = "train"

        #         cur_batch = cur_batch.to(DEVICE)
        #         print ("Current batch: ", cur_batch)
        #         print ("Current batch size: ", len(cur_batch))
        #         y1 = model(cur_batch)
        #         end_time = time.time()
        #         time_taken = end_time - start_time

        #         cur_N = cur_batch[0].x.size(0)
        #         print ("Current bucket size: ", cur_N, len(cur_batch))
        #         size_times[cur_N] = time_taken

        # pprint (size_times)
