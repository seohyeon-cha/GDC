proj_name=code_test_citeseer
display=inductive_1e-3_all
dataset=citeseer
nepochs=1700
wup=1e-5
id=0
runs=12
freq_path=pretrained_model/CITESEER/frequentist
path=pretrained_model/CITESEER/temp1e-5
# path=checkpoints/GDC_citeseer_GPU1_17h10m46s_on_Jul_21_2023/best_model_epoch_1588.pkl # semi-inductive
n_block=1
alpha=0.1

# bayes
bayes_citeseer_model_pth0=bayesian_mode_pths/bayes_citeseer_model_pth3.txt

# python main_bbgdc.py  --nblock $n_block --seed 41 --gpu_id 0 --wup $wup --do_bayes_cp --alpha $alpha --num_cal 500 --num_test 1000 --path $bayes_citeseer_model_pth0  --dataset $dataset --num_epochs $nepochs --num_run $runs 
seed_list=($(seq 101 1 200))
for index in $(seq 0 $((${#seed_list[@]} - 1)))
do
    seed=${seed_list[$index]}
    python main_bbgdc.py --do_bayes_cp --nblock $n_block --seed $seed --gpu_id $id --alpha $alpha --num_cal 500 --num_test 1000 --path $bayes_citeseer_model_pth0  --dataset $dataset --num_epochs $nepochs --num_run $runs 
done