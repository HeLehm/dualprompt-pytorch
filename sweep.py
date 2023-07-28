import wandb
import subprocess

def train():
    
    run = wandb.init()
    config = wandb.config

    #main.py cifar100_dualprompt --model vit_base_patch16_224 --batch-size 12 --data-path ./local_datasets/ --output_dir ./output

    command = f"python main.py --wandb cifar100_dualprompt --model vit_base_patch16_224 --lr {config.lr} --epochs {config.epochs} --silent True --use_mvn True"
    args = command.split()
    print()
    print(args)
    
    result = subprocess.run([*args, ], capture_output=True, text=True)
    result = float(result.stdout)


def main():
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'AA@1',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'values': [4,5,6]
            },
            'lr': {
                'max': 0.09, 'min': 0.003
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='dualprompt')

    # start python main.py
    wandb.agent(sweep_id, function=train)


if __name__ == '__main__':
    main()