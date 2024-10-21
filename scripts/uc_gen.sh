exp_name=$1
device=$2

export CUDA_VISIBLE_DEVICES=$device 
python ../generate.py --exp_name $exp_name