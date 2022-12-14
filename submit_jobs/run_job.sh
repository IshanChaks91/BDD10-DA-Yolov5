#!/bin/bash
#SBATCH --partition=gpu          # partition (queue)
#SBATCH --ntasks-per-node=64     # number of cores per node
#SBATCH --mem=128G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=0-12:00:00        # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=yolov5_bdd10k_baseyolov5pytorch_no_mosaic_seed450_test8_v1.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=yolov5_bdd10k_baseyolov5pytorch_no_mosaic_seed450_test8_v1.%j.err     # filename for STDERR hm  

# here comes the part with the description of the computational work, for example:
# load the OpenMPI environment
# source ~/.bashrc
# load cuda
# export PATH="~/anaconda3/condabin:$PATH"

source ~/.bashrc

conda init bash

conda deactivate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate BDD10-DA-Yolov5

#module load python3/3.8.1
module load cuda/11.2

cd ~/my_projects/BDD10-DA-Yolov5/BDD10-DA-Yolov5/yolov5

#python -V

python3 ~/my_projects/BDD10-DA-Yolov5/BDD10-DA-Yolov5/yolov5/train.py --img 1280 --batch 16 --epochs 300 --seed 450 --patience 50 \
	--hyp ~/my_projects/BDD10-DA-Yolov5/BDD10-DA-Yolov5/yolov5/data/hyps/hyp.scratch-low.yaml \
	--data /scratch/ichakr2s/yolov5pytorch/data.yaml \
	--weights yolov5s6.pt --cache
