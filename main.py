MARKET_PRICE = 2.5256

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

def action(buy, sell, gen, use, trade_price):
    a = (buy-sell) + gen
    b = a - use
    if a>=0:
        A = (buy-sell) * trade_price
    else:
        A = (-1) * gen * trade_price
    if b>=0:
        B = 0
    else:
        B = -1 * b * MARKET_PRICE
    return (A+B)

if __name__ == "__main__":
    args = config()
    data = []
    hour = []
    gen = []
    use = []

    # Predict tomorrow's consumption and generation data
    # TODO
    # Parameter needed later:
    # hour = list of dates with hours corresponding with 'gen' & 'use'
    # gen = prediction of next day's GENERATED electricity (list)
    # use = prediction of next day's CONSUMED electricity (list)

    # Decide to buy or sell
    for i in range(len(hour)):
        # normal_price = action(0, 0, gen[i], use[i], 0)
        sell_unit = round((0.9*gen), 2)
        sell_price = action(0, sell_unit, gen[i], use[i], sell_unit)
        data.append([hour[i], 'sell', sell_unit, sell_unit])

    output(args.output, data)