import pandas as pd
import glob
from sklearn.metrics import cohen_kappa_score

def perform_analysis():
    df1 = pd.read_csv("AnalysisData/annotator3.csv")
    df2 = pd.read_csv("AnalysisData/annotator4.csv")
    
    annotator_1 = df1['label']
    annotator_2 = df2['label']
    
    kappa = cohen_kappa_score(annotator_1, annotator_2)
    
    print(f"Cohen's Kappa: {kappa}")
    

def format():
    # Load the CSV file
    df = pd.read_csv('AnnotatedData/data3.csv')

    # Extract the integer value from the 'labels' column
    df['label'] = df['label'].str.extract(r'([-+]?\d+)').astype(int)

    # Save the modified DataFrame to a new CSV file
    df.to_csv('data3.csv', index=False)

    # Optionally, display the modified DataFrame
    print(df)

def extract():
    # Load the first CSV file and extract the first 302 rows
    df1 = pd.read_csv('AnnotatedData/data1.csv')
    df1_first_302 = df1.head(302)

    # Load the second CSV file and extract the first 130 rows
    df2 = pd.read_csv('AnnotatedData/data2.csv')
    df2_first_130 = df2.head(130)
    
    # Load the first CSV file and extract the first 302 rows
    df3 = pd.read_csv('AnnotatedData/data3.csv')
    df3_first_302 = df3.head(302)

    # Load the second CSV file and extract the first 130 rows
    df4 = pd.read_csv('AnnotatedData/data4.csv')
    df4_first_130 = df4.head(130)

    # Save the extracted rows into two new CSV files
    df1_first_302.to_csv('annotator1.csv', index=False)
    df3_first_302.to_csv('annotator2.csv', index=False)
    
    df2_first_130.to_csv('annotator3.csv', index=False)
    df4_first_130.to_csv('annotator4.csv', index=False)

def combineIntoOne():
    path = "AnnotatedData/"
    csv_files = glob.glob(path + "*.csv")
    dfs = [pd.read_csv(file) for file in csv_files]
    
    combined_df = pd.concat(dfs,ignore_index=True)
    
    combined_df.to_csv('combined_filed.csv', index=False)
    
    print(combined_df.head())
    
def main():
    # extract()
    perform_analysis()
    
if __name__ == "__main__":
    main()