device=$1

export CUDA_VISIBLE_DEVICES=$device 

python gt_train.py --exp_name uc_advbench0050 --save_name uc_advbench0050_e3 --epochs 3
python gt_train.py --exp_name uc_advbench0050 --save_name uc_advbench0050_e4 --epochs 4