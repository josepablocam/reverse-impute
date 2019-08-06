from argparse import ArgumentParser
import pandas as pd

def pivot(df, stats):
    pvt = pd.pivot_table(df, index="impute_method", values=stats, columns="approach")
    return pvt

def get_args():
    parser = ArgumentParser(description="Show results")
    parser.add_argument("--input", type=str, help="Results dataframe")
    parser.add_argument("--stats", nargs="+", type=str, help="Stats to show")
    return parser.parse_args()

def main():
    args = get_args()
    df = pd.read_csv(args.input)
    pvt = pivot(df, args.stats)
    print(pvt)

if __name__ == "__main__":
    main()
