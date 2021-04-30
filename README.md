# s21-team9-project

# Members
  - Hannah Williams
  - Gloria Abuka 
  - Jessica Wijaya
  - kirolous Shihataa 
  - Rober Makram


# Summary
  This project contains a Network that is trained to identify the age and gender of a person in an image. The age is set to 26 classifications, meaning it will output an estimated age range and not just a specific age. Datasets were over 400 GB couldn't include them in this repo, but can be downloaded from here https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ 
  
  
  
# Installation 
  1. Install all of the required libraries to run the python scripts.
      - pip install -r requirements.txt
      
  2. Download the final models
      - https://drive.google.com/drive/folders/1xsl6kqV_wdhqUZ26yHJ0Xs8h2C14qMqX
      
  3. Plug in the image you want to run the program on.
      - python Demo.py

# Files
    1. genderHistory.Json AgeHistory.json (files used to track the accuracy and loss of the models after each training set)
    2. imdb_outputdata.json wiki_outputdata.json (files that contain the metadata of the data sets "image locations, age, face location, etc... ")
    3. ProcessMatMetaData.ipynb (processed the mat files into the readable json files, minus all of the corrupted images from the datasets)
    4. TrainTheData.ipynb (notebook that contains all of functions and loading the model methods used to train the models, can be reused to train the final models even more)
  
 
  
    