import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score


src = "AnnotatedData/data4.csv"
target = "AnnotatedData/merged2.csv"
out = "AnnotatedData/merged2.csv"

def perform_cohen_analysis():
    df1 = pd.read_csv("Step3/AnalysisData/annotator1.csv")
    df2 = pd.read_csv("Step3/AnalysisData/annotator2.csv")
    
    df3 = pd.read_csv("Step3/AnalysisData/annotator3.csv")
    df4 = pd.read_csv("Step3/AnalysisData/annotator4.csv")
    
    annotator_1_for_data1 = df1['label']
    annotator_2_for_data3 = df2['label']
    annotator_3_for_data2 = df3['label']
    annotator_4_for_data4 = df4['label']
    
    dfs1 = [df1, df3]
    dfs2 = [df2, df4]
    
    combined_df_1 = pd.concat(dfs1,ignore_index=True)['label']
    combined_df_2 = pd.concat(dfs2,ignore_index=True)['label']
    
    kappa1 = cohen_kappa_score(annotator_1_for_data1, annotator_2_for_data3)
    kappa2 = cohen_kappa_score(annotator_3_for_data2, annotator_4_for_data4)
    kappa3 = cohen_kappa_score(combined_df_1, combined_df_2)
    
    print(f"Cohen's Kappa for data1.csv and data3.csv: {kappa1}")
    print(f"Cohen's Kappa for data2.csv and data4.csv: {kappa2}")
    print(f"Cohen's Kappa for BagA and BagB: {kappa3}")
    
def format2():
    path = "GPT/annotated_data7.csv"
    df = pd.read_csv(path)
    df['sentiment'] = df['label']
    df.to_csv(path, index=False)
    
def format():
    df = pd.read_csv("GPT/combined_filed.csv")
    
    extracted_df = df[['body', 'sentiment']]
    
    extracted_df.to_csv("combined_filed.csv", index=False)
    print(f"Extracted columns saved to combined_filed2.csv")

def extract():
    df1 = pd.read_csv('AnnotatedData/data1.csv')
    df1_first_302 = df1.head(302)

    df2 = pd.read_csv('AnnotatedData/data2.csv')
    df2_first_130 = df2.head(130)
    
    df3 = pd.read_csv('AnnotatedData/data3.csv')
    df3_first_302 = df3.head(302)

    df4 = pd.read_csv('AnnotatedData/data4.csv')
    df4_first_130 = df4.head(130)

    df1_first_302.to_csv('AnalysisData/annotator1.csv', index=False)
    df3_first_302.to_csv('AnalysisData/annotator2.csv', index=False)
    
    df2_first_130.to_csv('AnalysisData/annotator3.csv', index=False)
    df4_first_130.to_csv('AnalysisData/annotator4.csv', index=False)

def combineIntoOne():
    path = "GPT/"
    csv_files = glob.glob(path + "*.csv")
    dfs = [pd.read_csv(file) for file in csv_files]
    
    combined_df = pd.concat(dfs,ignore_index=True)
    
    combined_df.to_csv('GPT/combined_filed.csv', index=False)
    
    print(combined_df.head())
    
def perform_truth_table_analysis():
    file1 = "annotator3.csv"
    file2 = "annotator4.csv"
    output_file = "merged2.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    assert set(df1.columns) == set(df2.columns), "CSV files must have the same columns"

    merged_data = []

    for (_, row1), (_, row2) in zip(df1.iterrows(), df2.iterrows()):
        if row1['id'] != row2['id']:
            print(f"ID mismatch: {row1['id']} != {row2['id']}")
            continue

        if row1['label'] == row2['label']:
            merged_data.append(row1.tolist())
        else:
            print("\nConflict detected:")
            print(f"ID: {row1['id']}")
            print(f"Text: {row1['body']}")
            print(f"Label in File 1: {row1['label']}")
            print(f"Label in File 2: {row2['label']}")

            while True:
                try:
                    new_label = int(input("Enter the correct label (-1, 0, or 1): "))
                    if new_label in [-1, 0, 1]:
                        break
                    else:
                        print("Invalid input. Please enter -1, 0, or 1.")
                except ValueError:
                    print("Invalid input. Please enter an integer.")

            row1['label'] = new_label
            merged_data.append(row1.tolist())

    merged_df = pd.DataFrame(merged_data, columns=df1.columns)
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged CSV saved as {output_file}")

def extract_and_merge(source_file, target_file, output_file):
    source_df = pd.read_csv(source_file)
    # if len(source_df) < 131:
    #     print("Source file has less than 131 rows. Nothing to extract.")
    #     return
    extracted_df = source_df.iloc[130:] 
    
    target_df = pd.read_csv(target_file)
    
    merged_df = pd.concat([target_df, extracted_df], ignore_index=True)
    
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data written to {output_file}")

def display_distribution():
    df = pd.read_csv('Step3/FinalDataGPT/combined.csv')

    label_counts = df['sentiment'].value_counts().sort_index()
    print(label_counts)

    plt.figure(figsize=(6, 4))
    plt.bar(label_counts.index, label_counts.values, color=['red', 'blue', 'green'])
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Distribution of Labels')
    plt.xticks(label_counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
def get_normalized():
    df = pd.read_csv("Step3/FinalData/combined.csv")
    # Filter all rows where label is -1 or 1
    df_negative = df[df["label"] == -1]  # ≈ 400 rows
    df_positive = df[df["label"] == 1]   # ≈ 400 rows

    # Sample exactly 600 rows where label is 0
    df_neutral = df[df["label"] == 0].sample(n=300, random_state=42)

    # Combine the three subsets into a new dataframe
    df_balanced = pd.concat([df_negative, df_neutral, df_positive])

    # Shuffle the dataset to avoid order bias
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the new dataset
    df_balanced.to_csv("Step3/FinalData/normalized_data.csv", index=False)

    print(f"New dataset distribution:\n{df_balanced['label'].value_counts()}")
    print("Normalized dataset saved as 'normalized_data.csv'.")

def main():
    # perform_cohen_analysis()
    display_distribution()
    # get_normalized()
    # extract()
    # perform_truth_table_analysis()
    # extract_and_merge(src,target,out)
    # combineIntoOne()
    # format()
    
if __name__ == "__main__":
    main()