proj_name=CORA_inductive
display=frequentist
dataset=cora
id=1
runs=20
freq_path=pretrained_model/CORA/frequentist
alpha=0.1

# python main_freq.py --dataset $dataset --seed 41 --gpu_id $id --training
# python main_freq.py --dataset $dataset --seed 41 --gpu_id $id --wandb_save --wandb_proj_name $proj_name --display_name $display --training

# # inductive
# python main_freq_inductive.py --dataset $dataset --seed 41 --gpu_id $id --training
# python main_freq_inductive.py --num_epochs 2500 --dataset $dataset --seed 41 --gpu_id $id --wandb_save --wandb_proj_name $proj_name --display_name $display --training

# python main_freq.py --dataset $dataset --seed 41 --gpu_id $id --do_cp --alpha 0.1 --num_cal 500 --num_test 1000 --path $freq_path

# seed_list=($(seq 101 1 200))
# for index in $(seq 0 $((${#seed_list[@]} - 1)))
# do
#     seed=${seed_list[$index]}
#     python main_freq.py --dataset $dataset --seed $seed --gpu_id $id --do_cp --alpha $alpha --num_cal 500 --num_test 1000 --path $freq_path
# done
