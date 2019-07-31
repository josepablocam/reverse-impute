from argparse import ArgumentParser
import pickle

from training import TSDataset
from models import ReverseImputer


def add_predicted_mask_column(model, dataset, threshold=None):
    model = model.eval()
    yhat = model.predict_is_imputed(dataset.X, threshold=threshold)
    yhat = yhat.flatten(order="C")
    df = dataset.df
    df.predicted_mask = yhat
    return df


def get_args():
    parser = ArgumentParser(description="Add column with predicted imputed")
    parser.add_argument(
        "--model", type=str, help="Path to trained model weights"
    )
    parser.add_argument("--hidden_size", type=int, help="Model hidden size")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output csv with column predicting imputed status",
    )
    return parser.parse_args()


def main():
    args = get_args()
    model = ReverseImputer(args.hidden_size, args.hidden_size)
    model = model.load(args.model)
    with open(args.input, "rb") as fin:
        ts_data = pickle.load(fin)
    df = add_predicted_mask_column(model, df)
    df.to_csv(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
