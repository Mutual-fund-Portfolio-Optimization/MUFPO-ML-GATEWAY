def load_data(path):
    return pd.read_csv(path)

def set_date_to_datetime(df):
    df.date = pd.to_datetime(df.date)
    return df

def split_by_funds_name(x):
    return list(x.groupby('fund_name'))

def is_two_year_funds(fund):
    _, fund_df = fund
    time_difference =  fund_df.date.max() - fund_df.date.min()
    return time_difference >= pd.Timedelta(days=365 * 3)

class DataLoader:
    def __init__(self, path):
        self.path = path
    
    def filter_two_year_funds(self, x):
        return list(filter(is_two_year_funds, x))

    def __call__(self):
        return (
            Pipe(self.path)
            | load_data
            | set_date_to_datetime
            | split_by_funds_name
            | self.filter_two_year_funds
        ).get()
