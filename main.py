import os, argparse

file = """#!/bin/bash\n\
#\n\
#SBATCH --job-name=train_{a4}_{a3}_{a5}_{a8}\n\
#\n\
#SBATCH --cpus-per-task=2\n\
#SBATCH --ntasks=1\n\
#\n\
#SBATCH --mem-per-cpu=4G\n\
#SBATCH --partition=gpu\n\
#SBATCH --gres=gpu:TeslaV100:1\n\
#\n\
#SBATCH --mail-type='FAIL'\n\
#SBATCH --mail-user='manon.dausort@uclouvain.be'\n\
#SBATCH --output='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_{a4}_{a3}_{a5}_{a8}.out'\n\
#SBATCH --error='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_{a4}_{a3}_{a5}_{a8}.err'\n\

python3 train.py -com {a1} --num_epochs {a2} -m {a3} --wandb_b --model {a4} --lr {a5} --name_run test_{a4}_{a3}_{a5}_{a8} --freezed_bb {a7}
"""

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--com", type=str, default="no_comment")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--m", type=float, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument('--freezed_bb', type=int, default=1)
    
    parser.add_argument("--model_name",type=str, default="vgg16")
    
    args = parser.parse_args()
    
    if args.freezed_bb==1:
        freezed_backbone = 'fz_bb'
    else:
        freezed_backbone = 'ft_bb'
    
    with open("/CECI/home/users/m/d/mdausort/Cytology/slurm/submit_" + args.model_name + "_" + str(args.m) + "_" + str(args.lr) + "_" + freezed_backbone + ".sh","w") as writer:
        writer.write(file.format(a1=args.com, 
                                 a2=args.num_epochs,
                                 a3=args.m,
                                 a4=args.model_name,
                                 a5=args.lr,
                                 a6=args.bs, 
                                 a7=args.freezed_bb, 
                                 a8=freezed_backbone))
        writer.close()
    
    os.system("sbatch /CECI/home/users/m/d/mdausort/Cytology/slurm/submit_" + args.model_name + "_" + str(args.m) + "_" + str(args.lr) + "_" + freezed_backbone + ".sh")
