import os
if __name__ == "__main__":
    device = "cuda"
    base_cmd =  f"main.py --wandb cifar100_dualprompt --device {device} --batch-size 48 --data-path ./local_datasets/ --output_dir '' "
    cmds = [
        base_cmd , # like in paper
        base_cmd + ' --g_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11 --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11', # all prompt layers
        base_cmd + ' --e_prompt_layer_idx 0 1 2 3 4 5 6 7 8 9 10 11', # all e prompt layers
        base_cmd + ' --use_learnable_mask True', # learnable mask
        base_cmd + ' --use_learnable_mask True --learnable_mask_only_first_task True',
        base_cmd + ' --use_learnable_mask True --learnable_mask_binary True --learnable_mask_max_noise 0.5 --learnable_mask_g_binary_top_k 2 --learnable_mask_e_binary_top_k 3',
        base_cmd + ' --use_learnable_mask True --learnable_mask_binary True --learnable_mask_max_noise 0.5 --learnable_mask_g_binary_top_k 3 --learnable_mask_e_binary_top_k 5',
    ]

    # merge with both datasets 

    cmds_1 = []
    for cmd in cmds:
        cmd1 = cmd + ' --dataset "Split-CIFAR100"'
        cmd2 = cmd + ' --dataset "Split-CUB200"'
        cmds_1.append(cmd1)
        cmds_1.append(cmd2)

    # rotate seed
    cmds_2 = []
    for cmd in cmds_1:
        for seed in range(3):
            cmd1 = cmd + f' --seed {seed}'
            cmds_2.append(cmd1)

    # run
    for cmd in cmds_2:
        print(cmd)
        os.system(cmd)