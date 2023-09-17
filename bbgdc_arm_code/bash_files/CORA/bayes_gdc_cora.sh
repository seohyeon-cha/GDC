proj_name=code_test_cora
display=frequentist_cora
dataset=cora
id=0
runs=20
freq_path=pretrained_model/CORA/frequentist
path=pretrained_model/CORA/temp1e-5
wup=1e-5
# freq_path=checkpoints/GCN_cora_GPU0_13h19m32s_on_Jul_21_2023/best_model_epoch_1512.pkl # fully-inductive
# freq_path=checkpoints/GCN_cora_GPU0_16h09m25s_on_Jul_21_2023/best_model_epoch_1512.pkl # semi-inductive
# path=checkpoints/GDC_cora_GPU0_16h14m34s_on_Jul_21_2023/best_model_epoch_1894.pkl
n_block=1
alpha=0.1

# bayes
model_path=bayesian_mode_pths/bayes_cora_model_pth1.txt

# python main_bbgdc.py  --nblock $n_block --seed 41 --gpu_id 0 --wup $wup --do_bayes_cp --alpha $alpha --num_cal 500 --num_test 1000 --path $bayes_citeseer_model_pth0  --dataset $dataset --num_run $runs 
seed_list=($(seq 101 1 200))
for index in $(seq 0 $((${#seed_list[@]} - 1)))
do
    seed=${seed_list[$index]}
    python main_bbgdc.py --do_bayes_cp --nblock $n_block --seed $seed --gpu_id $id --alpha $alpha --num_cal 500 --num_test 1000 --path $model_path  --dataset $dataset --num_run $runs 
done