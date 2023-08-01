import subprocess

def main():
    #start subpreocess

    seeds = [42, 1999, 666]
    use_mvn = [True, False]

    for seed in seeds:
        for mvn in use_mvn:
            cmd = f'python main.py --wandb imr_dualprompt --seed {seed} --epochs 10 --lr 0.02'

            if mvn:
                cmd += ' --use_e_mvn True'
            
            print()
            print("Running: ", cmd)
            print()

            subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    main()