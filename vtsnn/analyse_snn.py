"""Runs analyses on a trained VT-SNN model.

Here, we plot the confusion matrices, and the early classification curves for the trained model."""

import argparse
import os
import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.legend import Legend

from torch.utils.data import DataLoader
import slayerSNN as snn
from vtsnn.models.snn import SlayerMLP, SlayerMM
from vtsnn.dataset import ViTacDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--runs", action="store", type=str, help="Path containing all the run directories.", required=True
)

run_args = parser.parse_args()

mode_map = {
    "tact": "Tactile",
    "vis": "Vision",
    "mm": "Combined"
}

def _is_run_dir(p):
    file_list = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    return ("args.pkl" in file_list)

def plot_confusion(predicted, actual, model, args):
    "Plots the confusion matrix."
    fig, ax = plt.subplots(figsize=(6,4))
    data = { "predicted": predicted, "actual": actual}
    df = pd.DataFrame(data, columns=["predicted", "actual"])
    cfm = pd.crosstab(df["actual"], df["predicted"], rownames=["Actual"], colnames=["Predicted"])
    sns.heatmap(cfm, annot=True)
    output_dir = f"confusion/{args.task}/{args.mode}_{args.loss}_{args.sample_file}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / f"{model.name}.png", tight_layout=True)

def analyse_model(model_dir):
    "Saves the confusion matrix, and returns a Panda dataframe for the accuracy."
    device = torch.device("cuda")
    net = torch.load(model_dir / "model_500.pt").to(device)
    pickled_args = Path(model_dir / "args.pkl")
    with open(pickled_args, "rb") as f:
        args = pickle.load(f)

    if args.task == "cw":
        output_size = 20
    else:
        output_size = 2

    test_dataset = ViTacDataset(
        path=args.data_dir,
        sample_file=f"test_80_20_{args.sample_file}.txt",
        output_size=output_size,
        spiking=True,
        mode=args.mode,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
    )

    all_outputs = []
    predictions = []
    actual = []

    correct = 0
    accs = {
        "Accuracy": [],
        "Length": [],
        "Modality": [],
        "Loss Function": [],
        "Model": [],
        "Args": []
    }

    net.eval()
    with torch.no_grad():
        for *data, target, label in test_loader:
            data = [d.to(device) for d in data]
            target = target.to(device)
            output = net.forward(*data)
            prediction = snn.predict.getClass(output)
            correct += torch.sum(
                prediction == label
            ).data.item()
            predictions.extend(prediction)
            actual.extend(label)
            all_outputs.append([label, output])

        for t in range(output.shape[-1]):
            early_correct = 0
            for label, output in all_outputs:
                early_output = output[...,:t]
                early_correct += torch.sum(snn.predict.getClass(early_output) == label).data.item()
            accs["Accuracy"].append(early_correct / len(test_loader.dataset))
            accs["Model"].append(model_dir.name)
            accs["Modality"].append(mode_map.get(args.mode))
            accs["Loss Function"].append(args.loss)
            accs["Length"].append(t)
            accs["Args"].append(str(args))

    predictions = list(map(lambda v: v.item(), predictions))
    actual = list(map(lambda v: v.item(), actual))

    plot_confusion(predictions, actual, model_dir, args)

    return pd.DataFrame(accs)

if __name__ == "__main__":
    dfs = []
    runs = [os.path.join(run_args.runs, dir) for dir in os.listdir(run_args.runs)
            if os.path.isdir(os.path.join(run_args.runs, dir))]
    for model in runs:
        if _is_run_dir(model):
            print(f"Processing {model}...")
            model_dir = Path(model)
            df = analyse_model(model_dir)
            dfs.append(df)

    combined_df = pd.concat(dfs)
    combined_df = combined_df.assign(t=combined_df.Length*0.02)
    combined_df.to_pickle("df.pkl")

    sns.set_context('paper')
    sns.set_style('whitegrid')
    sns.set(rc={"xtick.bottom" : True, 'font.sans-serif': 'Liberation Sans'}, context='paper', style='whitegrid', palette='Set1')
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(x="t", y="Accuracy", hue="Modality", style="Loss Function", ci="sd", linewidth=2, hue_order=["Tactile", "Vision", "Combined"], data=combined_df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    leg = Legend(ax, handles=handles[1:4], labels=labels[1:4], loc='upper left', frameon=True, prop={'size': 14})
    ax.add_artist(leg)
    leg = Legend(ax, handles=handles[5:], labels=labels[5:], loc='lower right', frameon=True, prop={'size': 14})
    ax.add_artist(leg)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    fig.savefig("early_classification.png", tight_layout=True)
