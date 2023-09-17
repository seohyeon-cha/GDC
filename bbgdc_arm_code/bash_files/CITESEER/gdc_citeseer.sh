proj_name=CITESEER_adaptive
display=sigmoid_1e-2_b2
dataset=citeseer
nepochs=2500
wup=1e-2
id=3
runs=12
freq_path=pretrained_model/CITESEER/frequentist
path=pretrained_model/citeseer_b2_sigmoid_0914/temp1e-2_b2
# path=checkpoints/GDC_citeseer_GPU1_17h10m46s_on_Jul_21_2023/best_model_epoch_1588.pkl # semi-inductive
n_block=2
alpha=0.1
# python main_freq.py --dataset $dataset --num_epochs $nepochs --seed 41 --gpu_id $id --training
# python main_freq.py --dataset $dataset --num_epochs $nepochs --seed 41 --gpu_id $id --wandb_save --wandb_proj_name $proj_name --display_name $display --training

# python main_freq.py --dataset $dataset --num_epochs $nepochs --seed 41 --gpu_id $id --do_cp --alpha $alpha --num_cal 500 --num_test 1000 --path $freq_path
# 
# seed_list=($(seq 201 1 300))
# for index in $(seq 0 $((${#seed_list[@]} - 1)))
# do
#     seed=${seed_list[$index]}
#     python main_freq.py --dataset $dataset --num_epochs $nepochs --seed $seed --gpu_id $id --do_cp --alpha $alpha --num_cal 500 --num_test 1000 --path $freq_path
# done


# python main_bbgdc_inductive.py --nblock $n_block --gpu_id $id --wup $wup  --dataset $dataset --num_epochs $nepochs --num_run $runs --training --wandb_save --wandb_proj_name $proj_name --display_name $display --seed 41 
# python main_bbgdc_inductive.py --nblock $n_block --gpu_id 0 --wup $wup --seed 41 --training  --dataset $dataset --num_epochs $nepochs --num_run $runs 

# python main_bbgdc.py  --nblock $n_block --seed 41 --gpu_id 0 --wup $wup --do_cp --alpha $alpha --num_cal 500 --num_test 1000 --path $path  --dataset $dataset --num_epochs $nepochs --num_run $runs 

# python main_bbgdc.py --nblock $n_block --gpu_id $id --wup $wup  --dataset $dataset --num_epochs $nepochs --num_run $runs --training --wandb_save --wandb_proj_name $proj_name --display_name $display --seed 41 
# python main_bbgdc.py --nblock $n_block --gpu_id 0 --wup $wup --seed 41 --training  --dataset $dataset --num_epochs $nepochs --num_run $runs 


seed_list=($(seq 101 1 200))
for index in $(seq 0 $((${#seed_list[@]} - 1)))
do
    seed=${seed_list[$index]}
    python main_bbgdc.py --nblock $n_block --seed $seed --gpu_id $id --wup $wup --do_cp --alpha $alpha --num_cal 500 --num_test 1000 --path $path  --dataset $dataset --num_epochs $nepochs --num_run $runs 
done

