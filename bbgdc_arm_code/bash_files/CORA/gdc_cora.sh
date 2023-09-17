proj_name=CORA_inductive
display=ind_1e-5_b2
dataset=cora
id=1
runs=20
freq_path=pretrained_model/CORA/frequentist
path=pretrained_model/cora_block1_adaptive_0914/ada_temp1e-3_b1
wup=1e-5
# freq_path=checkpoints/GCN_cora_GPU0_13h19m32s_on_Jul_21_2023/best_model_epoch_1512.pkl # fully-inductive
# freq_path=checkpoints/GCN_cora_GPU0_16h09m25s_on_Jul_21_2023/best_model_epoch_1512.pkl # semi-inductive
# path=checkpoints/GDC_cora_GPU0_16h14m34s_on_Jul_21_2023/best_model_epoch_1894.pkl
n_block=1
alpha=0.1 

# python main_bbgdc.py --nblock $n_block --gpu_id $id --wup $wup  --dataset $dataset --num_run $runs --training --seed 41 
# python main_bbgdc.py --num_epochs 2500 --nblock $n_block --gpu_id $id --wup $wup  --dataset $dataset --num_run $runs --training --wandb_save --wandb_proj_name $proj_name --display_name $display --seed 41 

# python main_bbgdc_inductive.py --nblock $n_block --gpu_id $id --wup $wup  --dataset $dataset --num_run $runs --training --wandb_save --wandb_proj_name $proj_name --display_name $display --seed 41 
# python main_bbgdc_inductive.py --nblock $n_block --gpu_id 0 --wup $wup --seed 41 --training  --dataset $dataset --num_run $runs 

# python main_bbgdc.py --nblock $n_block --seed 41 --gpu_id 0 --wup $wup --do_cp --alpha 0.1 --num_cal 500 --num_test 1000 --path $path  --dataset $dataset --num_run $runs 

seed_list=($(seq 101 1 200))
for index in $(seq 0 $((${#seed_list[@]} - 1)))
do
    seed=${seed_list[$index]}
    python main_bbgdc.py --nblock $n_block --seed $seed --gpu_id $id --wup $wup --do_cp --alpha 0.1 --num_cal 500 --num_test 1000 --path $path  --dataset $dataset --num_run $runs 
done

