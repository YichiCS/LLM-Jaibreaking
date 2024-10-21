start=$1
end=$2
device=$3


export CUDA_VISIBLE_DEVICES=$device 
python filter.py --root /home/yichi/24Summer/UltraCOLD/results/uc_advbench0050 --start $start --end $end