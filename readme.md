# Named Entity Recognition using GPT-3.5-turbo
This project uses OpenAI's GPT-3.5-turbo to perform Named Entity Recognition on a dataset of job descriptions, extracting software tools and programming languages from the text. It then generates a word cloud based on the frequency of these entities and performs some additional analyses on the dataset.
## File Overview
1. `pull_data.py`: This script loads the initial data, filters it, and calls the GPT engine for each row to generate a DataFrame with additional columns for occupation, job duties, and job qualifications. The resulting DataFrame is saved as a pickle file.
2. `analyze_data.py`: This script loads the saved DataFrame from `pull_data.py` and performs Named Entity Recognition, cleaning, and processing. It also calculates some descriptive statistics and generates a word cloud based on the frequency of software tools and programming languages found in the job descriptions.
## Usage
1. Ensure you have the required Python libraries installed, including `pandas`, `openai`, `re`, `collections`, `itertools`, `wordcloud`, and `matplotlib`.
2. Replace the placeholder API key in `pull_data.py` with your actual OpenAI API key.
3. Run `pull_data.py` to load, filter, and process the data with GPT
## Outputs
1. ner_results.xlsx show the results of the script, including the text of the job ad and the entities pulled out
2. wordcloud.png is the wordcloud created with those entities.
