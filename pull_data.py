# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 21:02:56 2023

@author: abiga
"""
import pandas as pd
import openai
import time



# your API key)
with open("../key/key.txt", "r") as key_file:
    api_key = key_file.read().strip()
    openai.api_key = api_key

def loadFilterCleanData():
    df=pd.read_pickle("../data/all_cols_sample.pkl")
    yesDataSci=df.loc[df['occupation'].str[0:3]=="Yes"]
    return(yesDataSci)

def process_prompt(prompt, engine, temperature):
    """
    Processes a given prompt using the specified engine and temperature.

    Args:
        prompt (str): The input prompt.
        engine (str): The engine to be used for processing the prompt.
        temperature (float): The temperature to be used in processing the prompt.

    Returns:
        str: The response generated by the engine for the given prompt.
    """

    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=engine,
            messages=messages,
            max_tokens=1024,
            temperature=temperature)
        return response.choices[0]['message']['content']
    except Exception as e:
        print(
            f"Error processing prompt. Engine: {engine}, Prompt: {prompt}, Error: {str(e)[:100]}")
        return ''


def gpt_calls(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a sample DataFrame by calling the GPT engine for each row, generating a filtered DataFrame with additional columns for occupation, job duties, and job qualifications.

    Args:
        sample (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame with additional columns for occupation, job duties, and job qualifications.
    """

    engine = 'gpt-3.5-turbo'
    temperature = 0.1

    # Create empty lists to store results for both prompts
    results_prompt_1 = []

    # Iterate through the dataframe and process each prompt
    for _, row in sample.iterrows():
        time.sleep(2)

        # Process first prompt
        prompt_1 = f"I'd like you to please do named entity recognition on the following text and return one list with of software tools and programming languages: {row['info']}"

        response_1 = process_prompt(prompt_1, engine, temperature)
        results_prompt_1.append(response_1)

    sample['named_entities'] = results_prompt_1


    return sample

def main():
    yesDataSci=loadFilterCleanData()
    yesDataSci = gpt_calls(yesDataSci)
    yesDataSci.to_pickle("../data/entity_recognition.pkl")
    return(yesDataSci)
    
if __name__ == "__main__":
    cleaned_for_app=main()
    
