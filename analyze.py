# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 07:20:17 2023

@author: abiga
"""

import pandas as pd
import string
import re
from collections import Counter
from itertools import chain
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def gen_drop_list():
    drop_these = ["Software tools:", "Programming languages:",
                  "Software tools and programming languages:",
                  "None mentioned", "None identified", "None",
                  "USAJobs", "language codes", "in the text.", ".", "and",
                  'explicitly mentioned', 'explicitly']
    return(drop_these)


def clean_initial_string(s):
    drop_list = gen_drop_list()
    for i in drop_list:
        s = s.replace(i, " ")
    return s


def process_strings(string_list):
    processed_strings = []

    for s in string_list:
        s = s.strip()  # Remove leading and trailing spaces

        # Replace strings containing only punctuation or spaces with an empty string
        if all(c in string.punctuation + ' ' for c in s):
            s = ""

        # Split the string if there are two or more spaces
        parts = re.split(r'\s{2,}', s)

        # Extend the processed_strings list with the split parts
        processed_strings.extend(parts)

    return processed_strings


def normalize_string(s):
    s = s.lower()  # Convert to lowercase
    s = re.sub(f'[{string.punctuation}]', '', s)  # Remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()  # Replace multiple spaces with a single space and remove leading/trailing spaces
    return s


def strings_in_info(df):
    def check_strings(row):
        normalized_info = normalize_string(row['info'])
        normalized_string_list = [normalize_string(s) for s in row['more_clean']]

        # Get the strings in the list that are not present in the 'info' column
        missing_strings = [s for s in normalized_string_list if s not in normalized_info]

        # Return the missing strings, or an empty list if none are missing
        return missing_strings

    # Apply the check_strings function to each row and create a new column 'missing_strings'
    df['missing_strings'] = df.apply(check_strings, axis=1)

    return df


def count_items(df):
    counter = Counter()
    for row in df['more_clean']:
        counter.update([s for s in row if (s != '') & (s != "R") & (s != "Excel")])
    return(counter)


def find_missing_substrings(top_5, new_list):
    missing_items = []
    for item in top_5:
        found = False
        for new_item in new_list:
            if item in new_item:
                found = True
                break
        if not found:
            missing_items.append(item)
    return missing_items


def find_actually_missing(missing_items, info):
    actually_missing = []
    for item in missing_items:
        if item in info:
            actually_missing.append(item)
    return actually_missing


def remove_duplicates(item_list):
    return list(set(item_list))


def save_wordcloud(counter):
    # Replace empty lists with the string 'None'
    counter['None'] = counter.get('', 0)
    del counter['']

    # Create the word cloud object
    wordcloud = WordCloud(width=1500, height=1600, background_color='white', colormap='viridis')

    # Generate the word cloud using the frequencies
    wordcloud.generate_from_frequencies(counter)

    # Save the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png', dpi=300)  # Increase the DPI for higher text fidelity
    plt.show()

def main():
    df = pd.read_pickle("../data/entity_recognition.pkl")

    df['clean_named_entities'] = df['named_entities'].apply(clean_initial_string)
    df['clean_named_entities_list'] = df['clean_named_entities'].str.split(",")
    df['more_clean'] = df['clean_named_entities_list'].apply(process_strings)

    df['more_clean'] = df['more_clean'].apply(remove_duplicates)
    df = strings_in_info(df)

    counter = count_items(df)
    top_5 = [item[0] for item in counter.most_common(5)]

    df['missing_items'] = df['more_clean'].apply(lambda row: find_missing_substrings(top_5, row))
    df['actually_missing'] = df.apply(lambda row: find_actually_missing(row['missing_items'], row['info']), axis=1)

    # Drop the intermediate columns
    df.drop(columns=['clean_named_entities', 'clean_named_entities_list', 'missing_strings', 'missing_items', 'actually_missing'], inplace=True)

    # Calculate the median length of the 'more_clean' list for the subset where "Python" is one of the lists of strings
    python_subset = df[df['more_clean'].apply(lambda x: 'Python' in x)]
    median_length = python_subset['more_clean'].apply(len).median()
    print(f"The median length of the 'more_clean' list for the subset with 'Python': {median_length}")

    # Find the maximum number of strings in the 'more_clean' list and the corresponding PositionURI values
    max_strings = df['more_clean'].apply(len).max()
    max_strings_rows = df[df['more_clean'].apply(len) == max_strings]
    position_uris = max_strings_rows['PositionURI'].tolist()
    print(f"The maximum number of strings in the 'more_clean' list is {max_strings}.")
    print(f"The PositionURI values for rows with the maximum number of strings: {', '.join(position_uris)}")

    save_wordcloud(counter)
    return(df)

if __name__ == "__main__":
    df=main()

