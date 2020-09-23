import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER


def set_pin_domain(args):
    cpuinfo = get_cpuinfo()
    ppn = args.nproc_per_node
    cores_per_socket = int(cpuinfo['Core(s) per socket'])
    sockets = int(cpuinfo['Socket(s)'])
    total_cores = cores_per_socket * sockets
    cores_per_rank = total_cores // ppn
    pin_domain = "["
    for proc in range(ppn):
        domain_binary = 0
        begin = proc * cores_per_rank + args.ccl_worker_count
        end = proc * cores_per_rank + cores_per_rank -1 
        for i in range(begin, end + 1):
            domain_binary |= (1 << i)
        pin_domain += hex(domain_binary) + ","
    return pin_domain + "]"

def set_ccl_worker_affinity(args):
    cpuinfo = get_cpuinfo()
    ppn = args.nproc_per_node
    cores_per_socket = int(cpuinfo['Core(s) per socket'])
    sockets = int(cpuinfo['Socket(s)'] )
    total_cores = cores_per_socket * sockets
    cores_per_rank = total_cores // ppn
    affinity = ''
    #use the firt ccl_worker_count core of every rank for ccl communication 
    for proc in range(ppn):
        for ccl_worker in range(args.ccl_worker_count):
            affinity += str(proc * cores_per_rank + ccl_worker)+ "," 
    os.environ["CCL_WORKER_COUNT"] = str(args.ccl_worker_count)
    os.environ["CCL_WORKER_AFFINITY"] = affinity
    print("####CCL_WORKER_COUNT={}".format(os.environ["CCL_WORKER_COUNT"]))
    print("####CCL_WORKER_AFFINITY={}".format(os.environ["CCL_WORKER_AFFINITY"]))

def get_cpuinfo():
    origin_info = subprocess.check_output("lscpu", shell=True).strip().decode()
    info_list = origin_info.split("\n")
    info_dict = dict()
    for info in info_list:
        key_value = info.split(":")
        info_dict[key_value[0].strip()] = key_value[1].strip()
    return info_dict

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="Torch-ccl distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node")
    parser.add_argument("--ccl_worker_count", default=4, type=int,
                        help="core numbers per rank used for ccl communication")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")

    parser.add_argument("--hostfile", default="hostfile", type=str,
                         help="the file which store the host address list")
    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the training script"
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")
    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.hostfile):
        raise ValueError("{args.hostfile} not exist, Please create hostfile which include the ip list you used for distributed runing")
    # set distributed related environmental variables
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["CCL_ATL_TRANSPORT"] = 'ofi'
    mpi_pin_domain = set_pin_domain(args)
    cpuinfo = get_cpuinfo()
    ppn = args.nproc_per_node
    cores_per_socket = int(cpuinfo['Core(s) per socket'])
    sockets = int(cpuinfo['Socket(s)'] )
    total_cores = cores_per_socket * sockets
    cores_per_rank = total_cores // ppn

    print("####MPI_PIN_DOMAIN={}".format(mpi_pin_domain))
    set_ccl_worker_affinity(args)
    cmd = ['mpiexec.hydra']
    mpi_config = "-l -np {} -ppn {} -hostfile {}  -genv I_MPI_PIN_DOMAIN={} -genv OMP_NUM_THREADS={} ".format(args.nnodes*args.nproc_per_node,
                  args.nproc_per_node, args.hostfile, mpi_pin_domain, cores_per_rank - args.ccl_worker_count)
    cmd.extend(mpi_config.split())
    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)
    print("####run comand", cmd)
    process = subprocess.Popen(cmd, env=os.environ)
    process.wait()
 
if __name__ == "__main__":
    main()

