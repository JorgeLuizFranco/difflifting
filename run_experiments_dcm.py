
import subprocess
import os
from colorama import Fore, Style, init
import numpy as np
#Must be imported first
import mkl
init(autoreset=True)

def check_dataset_args(dataset):
    if dataset == "Cora":
        dataset_args = ["--lr", str(0.01),
                        "--hidden_dim", str(64),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    elif dataset == "Citeseer":
        dataset_args = ["--lr", str(0.01),
                        "--hidden_dim", str(128),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    elif dataset == "Pubmed":
        dataset_args = ["--lr", str(0.01),
                        "--hidden_dim", str(128),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    # elif dataset == "Texas" or dataset=="Wisconsin":
    #     dataset_args = ["--lr", str(0.001),
    #                     "--hidden_dim", str(64),
    #                     "--num_layers", str(2),
    #                     "--dataset", dataset,
    #                     "--weight_decay",str(5e-6)
    #                 ]
    elif dataset == "Texas" :
       dataset_args = ["--lr", str(0.01),
                       "--weight_decay", str(0),
                       "--batch_size", str(1),
                       "--hidden_dim", str(128),
                        "--num_layers", str(1),
                        "--dataset", dataset,

              ]
    elif dataset=="Wisconsin":
       dataset_args = ["--lr", str(0.01),
                       "--weight_decay", str(0),
                       "--batch_size", str(1),
                       "--hidden_dim", str(128),
                       "--num_layers", str(1),
                       "--dataset", dataset,

              ]
    elif dataset == "CS" or dataset== "Physics":
        dataset_args = ["--lr", str(0.001),
                        "--hidden_dim", str(128),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    elif dataset == "chameleon" or dataset=="squirrel":
        dataset_args = ["--lr", str(0.001),
                        "--hidden_dim", str(64),
                        "--num_layers", str(2),
                        "--dataset", dataset,
                        ]
    return dataset_args
# Lista dos valores de tnn e seed a serem testados
tnn_options = ["CWN"]
seed_options = [42, 3, 9]
datasets = ['Texas',"Wisconsin","Cora","Citeseer","chameleon" ]
# datasets = [ "NCI1", "NCI109", "MUTAG", "PROTEINS", "ZINC"]
gnn = "gin"
embedding_dims=[64]
num_layers_gnn=[2, 3]
k_max_values=[3, 5, 7, 9, 11]
# t_values = [round(x, 2) for x in np.arange(0.1, 10.01, 0.5)]


# Caminho para o script principal
script_path = "main.py"

# Garante que a pasta de logs exista
os.makedirs("logs", exist_ok=True)
venv_python = "python3"
for dataset in datasets:
    for seed in seed_options:
        logdir = f"DCM/results_hyperparameter_search/DCM_CWN_adaptk_v_kmax/{dataset}_{gnn}_32_2_{seed}"
        log_path = f"logs/CWN_DCM_seed{seed}.txt"
        print(Fore.CYAN + f"\n[RUNNING] --tnn CWN, --seed {seed}\n{'-' * 50}")

        other_configs = [
                        "--lifting", "DCMLifting",
                        "--logdir", logdir,
                        "--tnn", "CWN",
                        "--seed", str(seed)]
        cmd = [
            venv_python,
            script_path,


        ] + check_dataset_args(dataset) + other_configs

        with open(log_path, "w",  encoding="utf-8") as log_file:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,

                                       encoding='utf-8',
                                       errors='replace'
                                       )

            for line in process.stdout:
                print(Fore.WHITE + line, end="")  # imprime colorido
                log_file.write(line)

            process.wait()

            if process.returncode == 0:
                print(Fore.GREEN + f"\n[SUCCESS] Finalizado com sucesso para --tnn CWN, --seed {seed}")
            else:
                print(Fore.RED + f"\n[ERROR] Código de retorno {process.returncode} para --tnn CWN, --seed {seed}")
                print(Fore.YELLOW + f"[LOG] Veja detalhes em: {log_path}")
