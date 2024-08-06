import pandas as pd
import csv
from helpe import *

df = pd.read_csv("lender_list.csv", usecols=["Lender"])
df.dropna(subset=['Lender'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv("lender.csv",header=True, columns=df, index=False)
# df = df[:2000]


# adding in features
df = parse(df)

# adding in target 
df['target'] = df.apply(target, axis=1)


filtered_df_1 = df[df["target"] == 1]
filtered_df = df[df["target"] == 0]

filtered_df.to_csv("lender_company_grantee.csv",header=False, index=False)
filtered_df_1.to_csv("lender_name.csv",header=True, index=False)

# combine the csv files 
df1 = pd.read_csv('lender_company_grantee.csv', header=None)
df2 = pd.read_csv('lender_name.csv', header=None)
# df3 = pd.read_csv('lender_none.csv', header=None)

# Concatenate DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

combined_df.columns = ["Lender","two_words","contains_keyword","char_count","digit_count","special_char_count","uppercase_word_count","title_case_word_count","avg_word_length","contains_single_letter","target"]

# Write to a new CSV file
combined_df.to_csv('lender.csv', index=False, header=True)
df = pd.read_csv("lender.csv")