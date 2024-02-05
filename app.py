import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#------------------------------STREAMLIT APP--------------------------------

st.title("House data Prediction App")

#load the datasetset

dataset = pd.read_excel("HouseData (1).xlsx")

st.image("photoholgic-fn6x1TL290w-unsplash.jpg")

st.caption("House datasetset Explorer and RandomForest Classifier Model")

#Add the header

st.header("Dataset Concept.", divider="blue")

#Add paragrawaterfront explaining the datasetset

st.write("""The dataset contains critical information about 
         residential properties, such as transaction details, pricing, bedrooms, bathrooms, 
         living space, lot size, floors, and other property characteristics. 
         It provides a comprehensive view of each property's characteristics, including condition, grade, 
         waterfront status, and view. This datasetset, which includes dataset on construction year, location, and geograwaterfrontical coordinates, 
         is a valuable resource for real estate professionals and analysts, providing insights into market trends and 
         property values that can be used to make informed decisions.""")

#--------------------------EDA---------------------------------

st.header("Exploratory dataset Analysis", divider="blue")

if st.checkbox("dataset info"):
     st.write("datasetset info", dataset.info())
     
if st.checkbox("Number of Rows"):
     st.write("Number of Rows", dataset.shape[0])
     
if st.checkbox("Number of Columns"):
     st.write("Number of Columns", dataset.columns.tolist())
     
if st.checkbox("dataset types"):
     st.write("dataset types", dataset.dtypes)
     
if st.checkbox("Missing Values"):
     st.write("Missing Values", dataset.isnull().sum())
     
if st.checkbox("Statistical Summary"):
     st.write("Statistical Summary", dataset.describe())


#-----------------------VISALISATION--------------------

st.header("Visualization of the datasetset", divider='blue')

if st.button("Generate Bar Chart"):
    selected_columns = st.multiselect("Select the columns to visualize the Bar Chart", dataset.columns)
    #plot a bar chart
    if selected_columns:
        st.bar_chart(dataset[selected_columns])
    else:
        st.warning("please select at least two columns.")
    
        

# Button for line chart
if st.button("Generate Line Chart"):
    # Plot a line chart
    selected_columns = st.multiselect("Select the columns to visualize the Line Chart", dataset.columns)
    if selected_columns:
        for column in selected_columns:
            st.line_chart(dataset[column])
    else:
        st.warning("Please select at least one column.")

#prepare the dataset

X=dataset.drop("dataset", axis = 1)
y=dataset["dataset"]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=42)

#create and fit model

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)

#user input

st.sidebar.header("Slide for values")
id = st.sidebar.slider("ID", (dataset['id']).min(),(dataset['id']).max(),(dataset["id"]).mean())

date = st.sidebar.slider("Date", (dataset['date']).min(),(dataset['date']).max(),(dataset["date"]).mean())

price = st.sidebar.slider("Price", (dataset['price']).min(),(dataset['price']).max(),(dataset["price"]).mean())

bedrooms = st.sidebar.slider("Bedrooms", (dataset['bedrooms']).min(),(dataset['bedrooms']).max(),(dataset["bedrooms"]).mean())

bathrooms = st.sidebar.slider("Bathrooms", (dataset['bathrooms']).min(),(dataset['bathrooms']).max(),(dataset["bathrooms"]).mean())

sqft_living = st.sidebar.slider("Sqft living", (dataset['sqft living']).min(),(dataset['sqft living']).max(),(dataset["sqft living"]).mean())

sqft_lot = st.sidebar.slider("Sqft lot", (dataset['sqft lot']).min(),(dataset['sqft lot']).max(),(dataset["sqft lot"]).mean())

floors = st.sidebar.slider("Floors", (dataset['floors']).min(),(dataset['floors']).max(),(dataset["floors"]).mean())

waterfront = st.sidebar.slider("Waterfront", (dataset['waterfront']).min(),(dataset['waterfront']).max(),(dataset["waterfront"]).mean())

view = st.sidebar.slider("View", (dataset['view']).min(),(dataset['view']).max(),(dataset["view"]).mean())

#predict button

if st.sidebar.button("Predict"):
    
    #create datasetframe
    
    user_input = pd.datasetFrame(
        {
            'id':[id],
            'date':[date],
            'price':[price],
            'bedrooms':[bedrooms],
            'bathrooms':[bathrooms],
            'sqft living':[sqft_living],
            'sqft lot':[sqft_lot],
            'floors':[floors],
            'waterfront':[waterfront],
            'view':[view],
        }
    )\
    
    #prediction of data of the houses
    
    prediction = rf.predict(user_input)
    
    #display of the prediction
    st.sidebar.subheader('prediction')
    st.sidebar.write(f"From the information provided the wine quality is{prediction[0]}")

