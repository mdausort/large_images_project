import os
import argparse

csv_file = """#!/bin/bash\n\
#\n\
#SBATCH --job-name=csv_{pw}{ph}_{sp}_{m}\n\
#\n\
#SBATCH --cpus-per-task=1\n\
#SBATCH --ntasks=8\n\
#\n\
#SBATCH --mem-per-cpu=8G\n\
#SBATCH --partition=cp3\n\
#\n\
#SBATCH --mail-type='FAIL'\n\
#SBATCH --mail-user='manon.dausort@uclouvain.be'\n\
#SBATCH --output='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_csv_{pw}{ph}_{sp}_{m}.out'\n\
#SBATCH --error='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_csv_{pw}{ph}_{sp}_{m}.err'\n\

python3 DBTA_csv_creation.py --csv_dir {csv_dir} --images_dir {images_dir} -pw {pw} -ph {ph} -sp {sp} -m {m} --set_percent {set_percent} 
"""

train_file = """#!/bin/bash\n\
#\n\
#SBATCH --job-name=train_{m}_{lr}_{bs}\n\
#\n\
#SBATCH --cpus-per-task=1\n\
#SBATCH --ntasks=16\n\
#\n\
#SBATCH --mem-per-cpu=4G\n\
#SBATCH --partition=gpu\n\
#SBATCH --gres=gpu:TeslaV100:1\n\
#\n\
#SBATCH --mail-type='FAIL'\n\
#SBATCH --mail-user='manon.dausort@uclouvain.be'\n\
#SBATCH --output='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_train_{m}_{lr}_{bs}.out'\n\
#SBATCH --error='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_train_{m}_{lr}_{bs}.err'\n\

python3 train.py --task {task} -pw {pw} -ph {ph} -sp {sp} -m {m} -com {com} --num_epochs {num_epochs} --momentum {momentum} --wandb_ --freezed_bb {freezed_bb} --model {model} --name_run {m}_{lr}_{bs}_{model}_{freezed_bb}
"""

test_file = """#!/bin/bash\n\
#\n\
#SBATCH --job-name=test_{m}_{lr}_{bs}\n\
#\n\
#SBATCH --cpus-per-task=16\n\
#SBATCH --ntasks=1\n\
#\n\
#SBATCH --mem-per-cpu=4G\n\
#SBATCH --partition=gpu\n\
#SBATCH --gres=gpu:TeslaV100:1\n\
#\n\
#SBATCH --mail-type='FAIL'\n\
#SBATCH --mail-user='manon.dausort@uclouvain.be'\n\
#SBATCH --output='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_test_{m}_{lr}_{bs}.out'\n\
#SBATCH --error='/CECI/home/users/m/d/mdausort/Cytology/slurm/slurmJob_test_{m}_{lr}_{bs}.err'\n\

python3 train.py --task {task} -pw {pw} -ph {ph} -sp {sp} -m {m} -com {com} {wandb_} --model {model} --name_run {m}_{lr}_{bs}_{model}_{freezed_bb}
"""

PATH = '/CECI/home/users/m/d/mdausort/Cytology/'
# PATH = "/auto/home/users/t/g/tgodelai/tl"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="train", choices=["csv", "train", "test"], help="Task to do")

    # create_csv.py arguments
    parser.add_argument("--csv_dir", type=str, default=None, help="Directory of the csv file")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory of the images")

    parser.add_argument("-pw", "--patch_w", type=int, default=416, help="Width of the patch")
    parser.add_argument("-ph", "--patch_h", type=int, default=416, help="Height of the patch")
    parser.add_argument("-sp", "--stride_percent", type=float, default=1.0, help="Stride percentage")
    parser.add_argument("--set_percent", type=float, nargs='+', default=[0.7, 0.1, 0.2], help="Distribution percentage")
    parser.add_argument("-m", "--magnification", type=float, default=20, help="Magnification level")
    # parser.add_argument("-p", "--patients", type=str, nargs='+', default=['adult'], choices=['adult', 'child'], help="Patient identifier")

    # train.py arguments
    parser.add_argument("--model_name", type=str, default="vgg16")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument('--freezed_bb', type=int, default=1)

    parser.add_argument("-com", "--comment", type=str, default="no comment", help="Specific comment on the run")
    # parser.add_argument("--wandb_", type=str, default='', choices=['', '--wandb_'], help="Use of wandb")
    # parser.add_argument("--zip_b", type=float, default = '', choices=['', '--zip_b'], help="If images are contained in zip files")

    args = parser.parse_args()
    if args.freezed_bb == 1:
        f_bb = 'fz_bb'
    else:
        f_bb = 'ft_bb'

    if args.task == 'csv':
        sh_file_name = f"csv_{args.pw}{args.ph}_{args.sp}_{args.m}.sh"
        with open(os.path.join(PATH, sh_file_name), "w") as writer:
            writer.write(csv_file.format(csv_dir=args.csv_dir,
                                         images_dir=args.images_dir,
                                         pw=args.patch_w,
                                         ph=args.patch_h,
                                         sp=args.stride_percent,
                                         m=args.magnification,
                                         set_percent=args.set_percent,
                                         # zip_b=args.zip_b
                                         ))
            writer.close()

    elif args.task == 'train':
        sh_file_name = 'Training/' + f"train_{args.bs}_{args.lr}_{args.num_epochs}_{args.momentum}_{args.model_name}_{f_bb}.sh"
        with open(os.path.join(PATH, sh_file_name), "w") as writer:
            writer.write(train_file.format(task=args.task,
                                           pw=args.patch_w,
                                           ph=args.patch_h,
                                           sp=args.stride_percent,
                                           m=args.magnification,
                                           model=args.model_name,
                                           # p=args.p,
                                           set_percent=args.set_percent,
                                           # zip_b=args.zip_b,
                                           com=args.comment,
                                           bs=args.bs,
                                           lr=args.lr,
                                           num_epochs=args.num_epochs,
                                           momentum=args.momentum,
                                           # wandb_=args.wandb_,
                                           freezed_bb=args.freezed_bb))
            writer.close()

    elif args.task == 'test':
        sh_file_name = 'Testing/' + f"test_{args.bs}_{args.lr}_{args.num_epochs}_{args.momentum}_{args.model_name}_{f_bb}.sh"
        with open(os.path.join(PATH, sh_file_name), "w") as writer:
            writer.write(train_file.format(task=args.task,
                                           pw=args.patch_w,
                                           ph=args.patch_h,
                                           sp=args.stride_percent,
                                           m=args.magnification,
                                           model=args.model_name,
                                           # p=args.p,
                                           set_percent=args.set_percent,
                                           # zip_b=args.zip_b,
                                           com=args.comment,
                                           bs=args.bs,
                                           lr=args.lr,
                                           num_epochs=args.num_epochs,
                                           momentum=args.momentum,
                                           # wandb_=args.wandb_,
                                           freezed_bb=args.freezed_bb))
            writer.close()

    os.system("sbatch " + os.path.join(PATH, sh_file_name))
