import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def preprocess(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df.melt(['time'], var_name='distance', value_name='Distance')
    return df


def main():
    df = preprocess('distances.csv')
    sns.lineplot(df, x='time', y='Distance', hue='distance')
    plt.show()


if __name__ == '__main__':
    main()
