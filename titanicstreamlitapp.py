#What to copy? Where to paste? How to cusomize it?
# import streamlit as st
# import pickle

# st.title("Welcome to Faisal's Prediction App :ship:") 
# st.image('titanic2.jpg')
# pickleIn = open('titanicpickle.pkl', 'rb')
# classifier = pickle.load(pickleIn)

# #Defining the Function Which Will Make the Prediction Using the Data that User Will Input
# def prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
#     prediction = classifier.predict([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
#     print(prediction)
#     return prediction

# def main():
#     st.title('Titanic Prediction App!')

#     #The Following Code Creates Textboxes in Which the User Will Enter the Data Required
#     #to Make the Prediction
#     Pclass = st.text_input("Passenger Class")
#     Sex = st.text_input("Sex")
#     Age = st.text_input("Age")
#     SibSp = st.text_input("SibSp")
#     Parch = st.text_input("Parch")
#     Fare = st.text_input("Fare")
#     Embarked = st.text_input("Embarked")
#     result = ""
#     #This Code Ensures that When the Button 'Predict' is Clicked, the Prediction Function
#     #Defined Above is Called to Make the Prediction and Store it in the Variable 'Result'
#     if st.button('Predict'):
#         #Convert Inputs to the Appropriate Types
#         Pclass = int(Pclass)
#         Age = float(Age)
#         SibSp = int(SibSp)
#         Parch = int(Parch)
#         Fare = float(Fare)
#         result = prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
#     st.success(f'This out is: {result}')

# main()

import streamlit as st
import pickle
import numpy as np

# Set the title and an image for the web app
st.title("Welcome to Faisal's Titanic Prediction App :ship:")
st.image('titanic2.jpg')

# Load the pre-trained model
with open('titanicpickle.pkl', 'rb') as pickle_file:
    classifier = pickle.load(pickle_file)

# Function to make predictions
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Convert categorical inputs to numerical values
    Sex = 1 if Sex.lower() == 'male' else 0
    Embarked = {'C': 0, 'Q': 1, 'S': 2}.get(Embarked.upper(), 2)
    
    # Combine all inputs into a numpy array for model prediction
    input_features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = classifier.predict(input_features)
    
    # Return a more descriptive result
    return "Survived" if prediction[0] == 1 else "Did not survive"

# Main function for the Streamlit app
def main():
    st.header('Enter Passenger Details for Survival Prediction')

    # Input fields with type validation and placeholders
    Pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
    Sex = st.selectbox("Sex", ['Male', 'Female'])
    Age = st.number_input("Age", min_value=0, max_value=100, value=25)
    SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    Fare = st.number_input("Fare Paid", min_value=0.0, max_value=1000.0, value=32.0)
    Embarked = st.selectbox("Port of Embarkation", ['C (Cherbourg)', 'Q (Queenstown)', 'S (Southampton)'])

    # Button to trigger the prediction
    if st.button('Predict'):
        # Extract the first letter for the Embarked value
        Embarked = Embarked[0]
        result = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
        st.success(f'The prediction is: {result}')

# Run the main function
if __name__ == '__main__':
    main()

