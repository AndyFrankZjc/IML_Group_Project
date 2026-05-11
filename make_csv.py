import pandas as pd

df = pd.read_csv("processed_data/application_train_processed.csv")

def extract_rows(df, number: int) -> None:
    df_head = df.head(number)
    df_head.to_csv(f"view_data/application_train_processed_{number}.csv", index= False)

def count_column_rows(df, column_name: str) -> None:
    counts = df[column_name].value_counts(dropna=False) # include N/A value
    total_instances = len(df[column_name]) # include N/A value
    print(counts)
    print(f"total_instances: {total_instances}")

if __name__ == "__main__":
    #count_column_rows(df, "OCCUPATION_TYPE")
    extract_rows(df, 1000)