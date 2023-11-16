
# Files and Folders

This repository contains the following files:
. Containerfile: is the definition of the docker image. 
. dataset folder: Contains a csv file with the dataset used for this challenge without any modification.
. EDA: Contains the epxploratory analysis I did in order to understand the data
    . src folder the R notebook with the code
    . doc folder contains th ml_clallenge.html wich is only a compiled viewable version of the R. You can use it if you dont want to open the R file
. models: its empyt. It is used the the training and scoring code to store the pickle files with the models
. src folder:
    . BUILD.md includes the steps and commands needed to create the image from scratch and run it
    . run_challenge.sh: sh script used to run the training and scoring python files. You dont need to run it, It runs automatically when you run the container 
    . training_lgbm.py: this is the python code that gets the data, transform it, perform the hyperparameter search and saves a pickle file with the model
    . scoring_lgbm.py: load the model file, samples 10 registers from the dataset and calculates R2, MAE and MSE. it generates a json file in the models folder with the metrics
    . training.py: It trains a baseline linear regresion model I used for comparison. It also includes some funcions that are used by the other python file.
    . Requirements.txt: Includes all the libraries needed for this model to train and execute



# Building  the image

To build the image from scratch, you just need to follow the steps of the BUILD.md file. 
You have to have docker or podman to be able to do it.
The build process copies all the files that are needed


# Runninng the container
As part of the container file, I included a script that automatically runs the training and the scoring. However, you can run them as many times as you want.
The excution is shown in the terminal, and the results of the scoring are sent to the json file above mentioned. 

# Limitations
This version of the challenge does not include a database. Meaning, it only uses data from the csv.
I had many problems initializing a mySQL database within the same docker image
Finally I made it but it's I only can fun it in podman, and I don't want to pospone the delivery any longer.
I'm happy to show you the image and the database code if we do a presentation of this challenge. I do have the code, but didn't inlcude it in this repo to make it easier to diggest


# EDA and findings

During the EDA I found some interestesing thins I would like to point:
    . smokers tends to pay much more charges than no smokers
    . Linear models achieve more than 75% of R2, which is very good.
    . There is some structure in the residuals of the linear regressions I made, which tells me that with another funcional (like polinomic) form whe could get even better results
    . The LGBM used got better predictions than any of the linear models variations (altough not very usefull to understand the data)
    . There is a very interesting pattern in the plot of age vs charges spliting by smoker condition. The condition splits very well a part of the samples (low range of chartes no smokers and high range of charges for smokers). I think maybe there is another feature (not indluded in this dataset) that would help improve all these models.

    


