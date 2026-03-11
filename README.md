# 🚗 Car Price Predictor

Hey there! Welcome to my Car Price Prediction project. 

If you've ever wondered how much your used car might sell for, this project is built to answer exactly that. It's a machine learning web application that takes in basic details about a car—like its model, year of purchase, kilometers driven, and fuel type—and gives you an estimated resale price in Pakistani Rupees (PKR), formatted cleanly into Lakhs and Crores.

## What does it do?
It uses a trained Machine Learning model to look at historical car data and find patterns. Once it learns how different factors (like age or kilometers driven) affect a car's value, it can predict the price for any car you ask it about. 

## How to use it on your computer

If you want to run this project yourself on your own computer, it's pretty simple! You'll just need Python installed.

**1. Install the required libraries**  
Open your terminal or command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

**2. Train the Machine Learning Model**  
Before the app can predict anything, it needs to learn from the data. Run this command to train the model:
```bash
python train.py
```
*What happens here?* This script cleans up our car data (`data/car_data.csv`), tests a few different algorithms (like Random Forest and Gradient Boosting), picks the absolute best one, and saves it in the `model/` folder for later use.

**3. Start the Web App!**  
Now for the fun part. Run this command:
```bash
streamlit run app/app.py
```
Your web browser will pop open automatically with the app running. Just type in the car details, click predict, and you'll get your estimated price!

## What's inside this folder?
Here's a quick tour of how the code is organized:
- **`app/app.py`**: The actual web interface you interact with. It's built with Streamlit.
- **`train.py`**: The script that does all the heavy lifting to train our AI model.
- **`data/`**: This is where our dataset lives (`car_data.csv`).
- **`model/`**: After you run `train.py`, your trained model (`model.pkl`) gets saved here so the web app can use it later.
- **`notebooks/`**: Contains some scripts for exploring the data and generating graphs (Exploratory Data Analysis).
- **`src/predict.py`**: A helper file that the web app uses to ask the main model for a prediction.

## Cool things I learned building this
- **Age matters more than the exact year**: Instead of looking at the year "2018", telling the model that the car is "8 years old" makes it much smarter and better at guessing.
- **Some models are way better than others**: Simple Line graphs (Linear Regression) were okay, but more complex models like Random Forests were amazingly accurate in guessing prices.
- **Dropping names**: The specific car name had too many variations and caused "noise" in the model, so our model groups them strategically.

Feel free to play around with the code, learn from it, and tweak it to your liking!
