import os, random, logging, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from fusemap import *
import scanpy as sc
from pathlib import Path
import copy

def main(args):
    seed_all(0)
    ### set up logging
    Path(args.output_save_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_save_dir)

    arg_dict = vars(args)
    dict_pd = {}
    for keys in arg_dict.keys():
        dict_pd[keys] = [arg_dict[keys]]
    pd.DataFrame(dict_pd).to_csv(args.output_save_dir  + "config.csv", index=False)
    logging.info("\n\n\033[95mArguments:\033[0m \n%s\n\n", vars(args))
    logging.info("\n\n\033[95mArguments:\033[0m \n%s\n\n", vars(ModelType))


    ### read data
    logging.info("\n\nReading data...\n")
    X_input = []
    file_names = [
        args.input_data_folder_path + f
        for f in os.listdir(args.input_data_folder_path)
        if os.path.isfile(os.path.join(args.input_data_folder_path, f))
    ]

    for ind, file_name_i in enumerate(file_names):
        X = sc.read_h5ad(file_name_i)
        if "x" not in X.obs.columns:
            try:
                X.obs["x"] = X.obs["col"]
                X.obs["y"] = X.obs["row"]
            except:
                try:
                    X.obs['x']=X.obsm['spatial'][:,0]
                    X.obs['y']=X.obsm['spatial'][:,1]
                except:                
                    raise ValueError(
                        "Please provide spatial coordinates in the obs['x'] and obs['y'] columns"
                    )
        X.obs["name"] = f"section{ind}"
        X.obs['file_name'] = file_name_i.split('/')[-1]
        X_input.append(X)
    kneighbor = ["delaunay"] * len(X_input)
    input_identity = ["ST"] * len(X_input)

    ### train model
    logging.info("\n\nTraining model...\n")
    if args.mode == "integrate":
        spatial_integrate(
            X_input, args, kneighbor, input_identity
                          )
    elif args.mode == "map":
        for i in range(len(X_input)):
            args_i=copy.copy(args)
            args_i.output_save_dir = args.output_save_dir + f"/{X_input[i].obs['file_name'].unique()[0]}/"
            spatial_map(
            [X_input[i]], args_i, [kneighbor[i]], [input_identity[i]]
                        )   
            
    else:
        raise ValueError(f"mode {args.mode} not recognized")


    logging.info("\n\nDone!")

    return


if __name__ == "__main__":
    args = parse_input_args()
    main(args)
