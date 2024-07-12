import os
import argparse

csv_file = """#!/bin/bash\n\
#\n\
#SBATCH --job-name=csv_{cond}_{pw}_{sp}_{m}\n\
#\n\
#SBATCH --cpus-per-task=1\n\
#SBATCH --ntasks=8\n\
#\n\
#SBATCH --mem-per-cpu=8G\n\
#SBATCH --partition=cp3\n\
#\n\
#SBATCH --mail-type='FAIL'\n\
#SBATCH --mail-user={email}\n\
#SBATCH --output='{output_p}csv_{cond}_{pw}_{sp}_{m}.out'\n\
#SBATCH --error='{output_p}csv_{cond}_{pw}_{sp}_{m}.err'\n\

python3 DBTA_csv_creation.py --csv_dir {csv_dir} --images_dir {images_dir} -pw {pw} -ph {ph} -sp {sp} -m {m} --condition {cond}
"""

train_file = """#!/bin/bash\n\
#\n\
#SBATCH --job-name=train_{model}_{cond}_{pw}_{m}_{lr}_{bs}\n\
#\n\
#SBATCH --cpus-per-task=1\n\
#SBATCH --ntasks=16\n\
#\n\
#SBATCH --mem-per-cpu=4G\n\
#SBATCH --partition=gpu\n\
#SBATCH --gres=gpu:TeslaV100:1\n\
#\n\
#SBATCH --mail-type='FAIL'\n\
#SBATCH --mail-user={email}\n\
#SBATCH --output='{output_p}train_{model}_{cond}_{pw}_{m}_{lr}_{bs}.out'\n\
#SBATCH --error='{output_p}train_{model}_{cond}_{pw}_{m}_{lr}_{bs}.err'\n\

python3 train_2.py --task {task} -pw {pw} -ph {ph} --condition {cond} -sp {sp} -m {m} -com {com} --num_epochs {num_epochs} --val_frequency {val} --momentum {momentum} --freezed_bb {freezed_bb} --model {model} --name_run {model}_{cond}_{pw}_{m}_{lr}_{bs} --wandb_
"""

test_file = """#!/bin/bash\n\
#\n\
#SBATCH --job-name=test_{model}_{cond}_{pw}_{m}_{lr}_{bs}\n\
#\n\
#SBATCH --cpus-per-task=16\n\
#SBATCH --ntasks=1\n\
#\n\
#SBATCH --mem-per-cpu=4G\n\
#SBATCH --partition=gpu\n\
#SBATCH --gres=gpu:TeslaV100:1\n\
#\n\
#SBATCH --mail-type='FAIL'\n\
#SBATCH --mail-user={email}\n\
#SBATCH --output='{output_p}test_{model}_{cond}_{pw}_{m}_{lr}_{bs}.out'\n\
#SBATCH --error='{output_p}test_{model}_{cond}_{pw}_{m}_{lr}_{bs}.err'\n\

python3 train.py --task {task} -pw {pw} -ph {ph} --condition {cond} -sp {sp} -m {m} -com {com} --model {model} --name_run {model}_{cond}_{pw}_{m}_{lr}_{bs} --wandb_
"""

if __name__ == "__main__":

    thyroid = True

    if thyroid:
        email_adress = 'manon.dausort@uclouvain.be'
        output_path = '/CECI/home/users/m/d/mdausort/Cytology/slurm/'
    else:
        email_adress = 'tiffanie.godelaine@uclouvain.be'
        output_path = '/CECI/home/users/t/g/tgodelai/slurm/'

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="train", choices=["csv", "train", "test"], help="Task to do")

    # create_csv.py arguments
    parser.add_argument("--csv_dir", type=str, default=None, help="Directory of the csv file")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory of the images")
    parser.add_argument("--condition", type=str, choices=['mean', 'variance'])

    parser.add_argument("-pw", "--patch_w", type=int, default=416, help="Width of the patch")
    parser.add_argument("-ph", "--patch_h", type=int, default=416, help="Height of the patch")
    parser.add_argument("-sp", "--stride_percent", type=float, default=1.0, help="Stride percentage")
    parser.add_argument("--set_percent", type=float, nargs='+', default=[0.7, 0.1, 0.2], help="Distribution percentage")
    parser.add_argument("-m", "--magnification", type=float, default=20, help="Magnification level")
    parser.add_argument("-p", "--patients", type=str, default=None, choices=[None, 'adult', 'child'], help="Patient identifier")

    # train.py arguments
    parser.add_argument("--model_name", type=str, choices=['resnet50', 'resnet18', 'vgg16'])
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument('--freezed_bb', type=int, default=1)
    parser.add_argument("--val_frequency", type=int, default=10, help="Number of epochs between validations")

    parser.add_argument("-com", "--comment", type=str, default="no comment", help="Specific comment on the run")
    # parser.add_argument("--wandb_", type=str, default='', choices=['', '--wandb_'], help="Use of wandb")
    # parser.add_argument("--zip_b", type=float, default = '', choices=['', '--zip_b'], help="If images are contained in zip files")

    args = parser.parse_args()
    if args.freezed_bb == 1:
        f_bb = 'fz_bb'
    else:
        f_bb = 'ft_bb'

    if args.task == 'csv':
        sh_file_name = f"csv_{args.condition}_{args.patch_w}_{args.stride_percent}_{args.magnification}.sh"
        with open(os.path.join(output_path + 'sh/', sh_file_name), "w") as writer:
            writer.write(csv_file.format(csv_dir=args.csv_dir,
                                         images_dir=args.images_dir,
                                         cond=args.condition,
                                         pw=args.patch_w,
                                         ph=args.patch_h,
                                         sp=args.stride_percent,
                                         m=args.magnification,
                                         set_percent=args.set_percent,

                                         email=email_adress,
                                         output_p=output_path,
                                         # zip_b=args.zip_b
                                         ))
            writer.close()

    elif args.task == 'train':
        sh_file_name = f"train_{args.model_name}_{args.condition}_{args.patch_w}_{args.magnification}_{args.lr}_{args.bs}.sh"
        with open(os.path.join(output_path + 'sh/', sh_file_name), "w") as writer:
            writer.write(train_file.format(task=args.task,

                                           pw=args.patch_w,
                                           ph=args.patch_h,
                                           cond=args.condition,
                                           sp=args.stride_percent,
                                           set_percent=args.set_percent,
                                           m=args.magnification,
                                           # p=args.p,

                                           model=args.model_name,
                                           bs=args.bs,
                                           lr=args.lr,
                                           num_epochs=args.num_epochs,
                                           momentum=args.momentum,
                                           freezed_bb=args.freezed_bb,
                                           val=args.val_frequency,

                                           com=args.comment,
                                           email=email_adress,
                                           output_p=output_path,
                                           # wandb_=args.wandb_,
                                           ))
            writer.close()

    elif args.task == 'test':
        sh_file_name = f"test_{args.model_name}_{args.condition}_{args.patch_w}_{args.magnification}_{args.lr}_{args.bs}.sh"
        with open(os.path.join(output_path + 'sh/', sh_file_name), "w") as writer:
            writer.write(test_file.format(task=args.task,

                                          pw=args.patch_w,
                                          ph=args.patch_h,
                                          cond=args.condition,
                                          sp=args.stride_percent,
                                          set_percent=args.set_percent,
                                          m=args.magnification,
                                          # p=args.p,

                                          model=args.model_name,
                                          bs=args.bs,
                                          lr=args.lr,
                                          num_epochs=args.num_epochs,
                                          momentum=args.momentum,
                                          freezed_bb=args.freezed_bb,
                                          val=args.val_frequency,

                                          com=args.comment,
                                          email=email_adress,
                                          output_p=output_path,
                                          # wandb_=args.wandb_,
                                          ))
            writer.close()

    os.system("sbatch " + os.path.join(output_path + 'sh/', sh_file_name))
