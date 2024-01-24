from mufpo.etl import Pipe

PERIOD = 365

def forward_fill(fund, column='nav/unit'):
    key, fund_df = fund
    fund_df[column] = fund_df[column].ffill()
    return (key, fund_df)

def backward_fill(fund, column='nav/unit'):
    key, fund_df = fund
    fund_df[column] = fund_df[column].bfill()
    return (key, fund_df)

def split_train_test__inner(fund, period):
    key, fund_df = fund
    train = fund_df[:-period]
    test = fund_df[-period:]
    return (key, train, test)

def split_train_test(fund, period):
    return split_train_test__inner(fund, period)

class Preprocessor:
    def __init__(self, data_loader, period):
        self.data_loader = data_loader
        self.period = period

    def map_forward_fill(self, x):
        return list(map(forward_fill, x))
    
    def map_backward_fill(self, x):
        return list(map(backward_fill, x))

    def map_split_train_test(self, x):
        return list(
            map(
                lambda x: split_train_test(x, self.period), x
            )
        )

    def __call__(self):
        return (
            Pipe(self.data_loader)
            | self.map_forward_fill
            | self.map_backward_fill
            | self.map_split_train_test
        ).get()