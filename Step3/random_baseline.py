import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def create_random_baseline(input_file, output_file):
    print(f"Loading data from {input_file}")

    df = pd.read_csv(input_file)

    # Create completely random labels from the set {-1, 0, 1}
    possible_labels = [-1, 0, 1]
    random_labels = np.random.choice(possible_labels, size=len(df))

    # Create a new dataframe with random labels
    random_df = df.copy()
    random_df["label"] = random_labels

    # Save the new dataframe to a CSV file
    random_df.to_csv(output_file, index=False)

    print(f"Random baseline dataset saved to {output_file}")


if __name__ == "__main__":
    input_file = "FinalData/combined.csv"
    output_file = "FinalData/random_baseline.csv"

    create_random_baseline(input_file, output_file)
