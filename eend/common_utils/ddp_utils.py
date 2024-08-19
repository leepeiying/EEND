import os
import torch.distributed as dist
import torch
import random
import socket

# def getoneNode():
#     nodelist = os.environ['SLURM_JOB_NODELIST']
#     nodelist = nodelist.strip().split(',')[0]
#     import re
#     text = re.split('[-\[\]]', nodelist)
#     if ('' in text):
#         text.remove('')
#     return text[0] + '-' + text[1] + '-' + text[2]


# def dist_init(host_addr, rank, local_rank, world_size, port=23456):
#     host_addr_full = 'tcp://' + host_addr + ':' + str(port)
#     dist.init_process_group("nccl", init_method=host_addr_full,
#                             rank=rank, world_size=world_size)
#     assert dist.is_initialized()


# def init_ddp(gpu_list='[0]'):
#     if isinstance(gpu_list, int):
#         gpu_list = [gpu_list]

#     if 'WORLD_SIZE' in os.environ:
#         # using torchrun to start:
#         # egs: torchrun --standalone --rdzv_endpoint=localhost:$PORT_k --nnodes=1 --nproc_per_node=2 train.py --config conf/config.yaml --gpu_list '[0,1]'
#         host_addr = 'localhost'
#         rank = int(os.environ['RANK'])
#         local_rank = int(os.environ['LOCAL_RANK'])
#         world_size = int(os.environ['WORLD_SIZE'])
#         gpu_id = int(gpu_list[rank])
#         # dist_init(host_addr, rank, local_rank,
#                   # world_size, port)
#         dist.init_process_group(backend='nccl')
#     elif 'SLURM_LOCALID' in os.environ:
#         # start process using slurm
#         host_addr = getoneNode()
#         rank = int(os.environ['SLURM_PROCID'])
#         local_rank = int(os.environ['SLURM_LOCALID'])
#         world_size = int(os.environ['SLURM_NTASKS'])
#         dist_init(host_addr, rank, local_rank,
#                   world_size, '2' + os.environ['SLURM_JOBID'][-4:])
#         gpu_id = local_rank
#     else:
#         # run locally with only one process
#         host_addr = 'localhost'
#         rank = 0
#         local_rank = 0
#         world_size = 1
#         gpu_id = int(gpu_list[rank])
#         dist_init(host_addr, rank, local_rank,
#                   world_size, 8888 + random.randint(0, 1000))


#     return gpu_id

def getoneNode():
    nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    import re
    text = re.split('[-\[\]]', nodelist)
    if '' in text:
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]

def find_free_port():
    # Find a free port for initializing the process group
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def dist_init(host_addr, rank, local_rank, world_size, port=None):
    if port is None:
        port = find_free_port()
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    dist.init_process_group("nccl", init_method=host_addr_full,
                            rank=rank, world_size=world_size)
    assert dist.is_initialized()

def init_ddp(gpu_list):
    if isinstance(gpu_list, int):
        gpu_list = [gpu_list]
    elif isinstance(gpu_list, str):
        gpu_list = list(map(int, gpu_list.split(',')))  # Convert comma-separated string to list of ints

    if 'WORLD_SIZE' in os.environ:
        host_addr = 'localhost'
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = gpu_list[local_rank]
        torch.cuda.set_device(gpu_id)
        dist_init(host_addr, rank, local_rank, world_size)
    elif 'SLURM_LOCALID' in os.environ:
        host_addr = getoneNode()
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        gpu_id = gpu_list[local_rank]
        torch.cuda.set_device(gpu_id)
        dist_init(host_addr, rank, local_rank, world_size)
    else:
        host_addr = 'localhost'
        rank = 0
        local_rank = 0
        world_size = 1
        gpu_id = gpu_list[local_rank]
        torch.cuda.set_device(gpu_id)
        dist_init(host_addr, rank, local_rank, world_size)

    return gpu_id