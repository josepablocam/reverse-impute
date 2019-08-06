import argparse
import quandl
import pandas as pd

from bs4 import BeautifulSoup
import requests
import tqdm

QUANDL_ACTIVE = False


def activate_quandl(key=None):
    global QUANDL_ACTIVE
    if QUANDL_ACTIVE:
        return
    if key is None:
        with open("quandl-key.txt", "r") as fin:
            key = fin.read().strip()
    print("Using quandl key: {}".format(key))
    quandl.ApiConfig.api_key = key
    QUANDL_ACTIVE = True


def get_sp500_tickers(website=None):
    if website is None:
        website = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    pg = requests.request("GET", website)
    assert pg.status_code == 200, "Unable to get website"
    soup = BeautifulSoup(pg.text, "html.parser")
    table = soup.find("table", {"class": "wikitable sortable"})

    header = table.findAll('th')
    if header[0].text.strip() != "Symbol":
        raise Exception("Can't parse wikipedia's table!")

    tickers = []
    rows = table.findAll("tr")
    records = []
    rows = table.findAll('tr')
    results = []
    for row in rows[1:]:
        fields = row.findAll('td')
        if fields:
            try:
                symbol = fields[0].text.strip()
                name = fields[1].text.strip()
                results.append({"ticker": symbol, "name": name})
            except:
                pass
    return pd.DataFrame(results)


def get_stock_prices(
        tickers, start_date, end_date, n=10, source="WIKI/PRICES", key=None
):
    activate_quandl(key)
    failed = []
    dfs = []
    for i in tqdm.tqdm(range(0, len(tickers) + n, n)):
        chosen_tickers = tickers[i:(i + n)]
        if len(chosen_tickers) == 0:
            continue
        try:
            data = quandl.get_table(
                source,
                ticker=chosen_tickers,
                qopts={'columns': ['ticker', 'date', 'adj_close']},
                date={
                    'gte': start_date,
                    'lte': end_date
                }
            )
            dfs.append(data)
        except quandl.QuandlError as err:
            print(err)
            failed.extend(chosen_tickers)
    df = pd.concat(dfs, axis=0)
    df = df.rename(columns={"ticker": "ts_id", "adj_close": "orig"})
    df = df.reset_index(drop=True)
    return df, failed


def get_args():
    parser = argparse.ArgumentParser(description="Save down SP500 prices")
    parser.add_argument("-o", "--output", type=str, help="Output path for csv")
    parser.add_argument("-k", "--key", type=str, help="Quandl key")
    return parser.parse_args()


def main():
    args = get_args()
    if args.key is not None:
        activate_quandl(args.key)
    sp500_tickers = get_sp500_tickers()
    tickers = sp500_tickers.ticker.values.tolist()
    df, failed = get_stock_prices(
        tickers,
        n=10,
        start_date='2014-12-31',
        end_date='2018-12-31',
    )
    if len(failed)> 0:
        print("Failed to collect data for {}".format(failed))
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
