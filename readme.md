## This is pdf bot

- put all the files in data folder
- set OPENAI_API_KEY enviroment variable with valid key, this is required to chat only, indexing is done using opensource model from hugging face
- you can change model from constants.py, for now I have kept model focused on the quality, if required you can change it to model focused on performance.
- in main.py, make build_index=True, and run that file
- that will read all the pdfs in data folder and create embedding of it in the db folder
- now you can ask any question based on data you have provided in the data folder.