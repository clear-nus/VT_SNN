"""Runs analyses on a trained VT-SNN model.

Here, we plot the confusion matrices, and the early classification curves for the trained model."""

import argparse
import pickle
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from torch.utils.data import DataLoader
import slayerSNN as snn
from vtsnn.models.snn import SlayerMLP, SlayerMM
from vtsnn.dataset import ViTacDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dirs", action="store", type=str, help="Path to trained model.", nargs="+", required=True
)

run_args = parser.parse_args()

def plot_confusion(predicted, actual, model):
    "Plots the confusion matrix."
    data = { "predicted": predicted, "actual": actual}
    df = pd.DataFrame(data, columns=["predicted", "actual"])
    cfm = pd.crosstab(df["actual"], df["predicted"], rownames=["Actual"], colnames=["Predicted"])
    sns.heatmap(cfm, annot=True)
    plt.savefig(f"confusion_{model.name}.png", tight_layout=True)

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
            accs["Length"].append(t)
            accs["Args"].append(str(args))

    predictions = list(map(lambda v: v.item(), predictions))
    actual = list(map(lambda v: v.item(), actual))

    plot_confusion(predictions, actual, model_dir)

    return pd.DataFrame(accs)

if __name__ == "__main__":
    dfs = []
    for model in run_args.model_dirs:
        model_dir = Path(model)
        df = analyse_model(model_dir)
        dfs.append(df)

    combined_df = pd.concat(dfs)
    combined_df = combined_df.assign(t=combined_df.Length*0.02)
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(x="t", y="Accuracy", data=combined_df)
    fig.savefig("early_classification.png", tight_layout=True)
