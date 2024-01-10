import numpy as np
import pickle as pkl
import streamlit as st

loaded_model = pkl.load(open('knn_model_trained.sav','rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
	input_data_as_numpy_array = np.asarray(input_data)
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
	print("testing:************",input_data)
	prediction = loaded_model.predict(input_data_reshaped)
	print(prediction)
	if (prediction[0] == 0):
	  return 'The person is not diabetic'
	else:
	  return 'The person is diabetic'

def main():
	#giving a title
	st.title('Diabetes Prediction')

	#Getting input data from the user
	Pregnancies = st.text_input('Number of Pregnancies')
	Glucose = st.text_input('Glucose Level')
	BloodPressure = st.text_input('BP Value')	
	SkinThickness = st.text_input('SkinThickness Value')
	Insulin = st.text_input('Insulin Level')
	BMI = st.text_input('BMI Value')
	DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
	Age = st.text_input('Age of the person')

	#code for prediction
	diagnosis = ''

	#creating a button for prediction

	if st.button('Diabetes Test Result'):
		Pregnancies = float(Pregnancies)
		Glucose = float(Glucose)
		BloodPressure = float(BloodPressure)
		SkinThickness = float(SkinThickness)
		Insulin = float(Insulin)
		BMI = float(BMI)
		DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
		Age = float(Age)
		diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

	st.success(diagnosis)

if __name__ == '__main__':
	main()



