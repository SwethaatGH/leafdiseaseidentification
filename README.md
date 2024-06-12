# Leaf Disease Identification
This web application is an amateur attempt at deploying a machine learning model that can aid in detecting leaf diseases.

## Techstack used
Python keras for CNN, flask API; HTML, CSS, Javascript, Jinja2 template for Frontend; numpy, pandas, matplotlib for usual data manipulation.

CNN model used here is a very simple sequential dense architecture with basic layers like Conv2D, MaxPooling2D and BatchNormalization, Flatten and Dropout mechanisms.

## How to use
1. Clone the project into your projectfolder
2. Ensure correct directory structure
3. Run leafdisease.py using command ```python3 leafdisease.py``` in the terminal after navigating to projectfolder.
4. Model will be saved in projectfolder.
5. Run app.py in the terminal using ```python3 app.py``` and visit ```http://127.0.0.1:5000``` to view website coded in index.html
   
## Directory Structure
projectfolder

|--leafdisease.py

|--app.py

|--my_model.h5

|--templates

&emsp;|--index.html

&emsp;|--decor.png

Uploads folder will get created within projectfolder.

## Frontend

![image](https://github.com/SwethaatGH/leafdiseaseidentification/assets/98175379/d1f7016d-9ec6-4085-9da5-67f888df861a)
![image](https://github.com/SwethaatGH/leafdiseaseidentification/assets/98175379/58c96e7b-6dd6-4968-b717-d65833ca182f)
![image](https://github.com/SwethaatGH/leafdiseaseidentification/assets/98175379/5e5bca86-4397-400c-958e-7ed8e1a258e4)

## Additional

![image](https://github.com/SwethaatGH/leafdiseaseidentification/assets/98175379/5942207d-69e1-4aed-ba0a-d4d93ad9353d)
![Screenshot 2024-06-11 at 11 40 21 PM](https://github.com/SwethaatGH/leafdiseaseidentification/assets/98175379/47866cc3-f587-49f4-97ec-c5d07ee9d3b0)
![Screenshot 2024-06-11 at 10 59 20 PM](https://github.com/SwethaatGH/leafdiseaseidentification/assets/98175379/0ef910db-3695-4b0f-9a40-c4a291e82511)

