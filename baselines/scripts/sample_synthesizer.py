# this script is used to train the synthesizer
import os
import argparse
from lib.commons import load_config
from lib.info import *


def update_config(config, model, dataset):
    """add model configure and dataset configure to args"""
    config["path_params"]["meta_data"] = "datasets/{0}/{0}.json".format(dataset)
    config["path_params"]["train_data"] = "datasets/{0}/train.csv".format(dataset)
    config["path_params"]["val_data"] = "datasets/{0}/val.csv".format(dataset)
    config["path_params"]["test_data"] = "datasets/{0}/test.csv".format(dataset)
    config["path_params"]["raw_data"] = "datasets/{0}/{0}.csv".format(dataset)

    config["path_params"]["loss_record"] = "exp/{0}/{1}/loss.csv".format(dataset, model)
    config["path_params"]["out_model"] = "exp/{0}/{1}/{1}.pt".format(dataset, model)
    config["path_params"]["out_data"] = "exp/{0}/{1}/inf_epsilon_{0}.csv".format(dataset, model)

    config["path_params"]["fidelity_result"] = "exp/{0}/{1}/fidelity_result.json".format(dataset, model)
    config["path_params"]["privacy_result"] = "exp/{0}/{1}/privacy_result.json".format(dataset, model)
    config["path_params"]["fidelity_train_result"] = "exp/{0}/{1}/fidelity_train_result.json".format(dataset, model)
    config["path_params"]["utility_result"] = "exp/{0}/{1}/utility_result.json".format(dataset, model)

    # set the absolute path
    for key, value in config["path_params"].items():
        config["path_params"][key] = os.path.join(ROOT_DIR, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="pretraining_pategan")
    parser.add_argument("--dataset", "-d", type=str, default="kitchen-complete-v0")
    parser.add_argument("--cuda", "-c", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--finetuning', action='store_true')


    args = parser.parse_args()

    # load template config
    model_config = "exp/base_config/{0}.toml".format(args.model)
    config_path = os.path.join(ROOT_DIR, model_config)
    config = load_config(config_path)

    # modify config according to dataset and model
    update_config(config, args.model, args.dataset)

    # dynamically import model interface
    synthesizer = __import__("synthesizer." + args.model, fromlist=[args.model])
    print("sampling {0} on {1}".format(args.model, args.dataset))
    synthesizer.sample(config)


if __name__ == "__main__":
    main()
