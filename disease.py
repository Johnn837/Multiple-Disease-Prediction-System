import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.preprocessing import image

#loading the models

st.set_page_config(
    page_title = "Multiple Disease Prediction System",
    page_icon = "healthcare.png",  # Using the Staff of Aesculapius symbo
)

diabetes = pickle.load(open("models/diabetes.sav", "rb"))
heart_disease = pickle.load(open("models/heart_disease_model.sav", "rb"))
parkinsons_disease = pickle.load(open("models/parkinsons_model.sav", "rb"))
# breast_cancer = pickle.load(open("models/breast_cancer_model.sav", "rb"))
lung_cancer = pickle.load(open("models/lung_cancer_model.pkl", "rb"))

#sidebar for navigation
with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",            
                           [
                            "🏠 Home","💉 Diabetes Prediction",
                            "❤️ Heart Disease Prediction",
                            "🧠 Parkinsons Disease Prediction",
                            "🎗️ Breast Cancer Prediction",
                            "🫁 Lung Cancer Prediction",
                            "🦟 Malaria Prediction",
                            "🤧 Pneumonia Prediction"],
                            icons = ["home","syringe","hert","brain","breast","lung","mosquito","fever"],
                           default_index = 0)

# Home Page
if selected == "🏠 Home":
    st.title("Welcome to Multiple Disease Prediction System")
    st.subheader("Step into our Health Hub, where we mix tech brilliance with a friendly touch. Get ready for a health journey like never before!")

    st.subheader("Our Mission:")
    st.markdown("🚀 Buckle up because our mission is to make health predictions as easy as sending a text. We’re here to bring the power of machine learning straight to your fingertips.")

    st.subheader("What We Predict:")
    st.markdown("💡 Curious minds, listen up! We've got predictions for seven health superheroes: cancer, diabetes, heart disease, kidney disease, liver disease, malaria (parasitized and uninfected), and pneumonia. Bet you didn’t know predicting your health could be this cool.")

    st.subheader("How It Works:")
    st.markdown("🎲 It’s like magic, but better! For most diseases, just throw in some numbers. For malaria and pneumonia, hit us with images. Sit back, relax, and watch the tech magic unfold.")

    st.subheader("User-Friendly:")
    st.markdown("🚀 Blast off with our user-friendly interface! Created with Streamlit, Python, and HTML, it's like having a personal health assistant – easy, breezy, and kind of awesome.")

    st.subheader("Boosting Healthcare Powers:")
    st.markdown("⚡️ Calling all healthcare heroes! We're not just an app; we're your sidekick. Boost your diagnostic powers, make decisions like a rockstar, and let's show the world what healthcare awesomeness looks like!")


#Diabetes Prediction Page:
# Diabetes Prediction Page
if selected == "💉 Diabetes Prediction":
    st.title("Diabetes Prediction using Machine Learning")

    # Display input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies", placeholder="Enter number of pregnancies")
        
    with col2:
        Glucose = st.text_input("Glucose Level", placeholder="70-126 mg/dL")
    
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value", placeholder="80-130 mm Hg")
    
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value", placeholder="Enter skin thickness value")
    
    with col2:
        Insulin = st.text_input("Insulin Level", placeholder="2-25 µU/mL")
    
    with col3:
        BMI = st.text_input("BMI Value", placeholder="18 - 31")
    
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value", placeholder="0.08 - 2.42")
    
    with col2:
        Age = st.text_input("Age of the Person", placeholder="Enter age")


    # Validation
    if Pregnancies and not Pregnancies.isdigit():
        st.error("Number of Pregnancies should be a valid number.")

    elif Pregnancies and (int(Pregnancies) < 0 or int(Pregnancies) > 10):
        st.error("Number of Pregnancies should be between 0 and 10.")
    
    if Glucose and (not Glucose.replace('.', '', 1).isdigit() or float(Glucose) < 70 or float(Glucose) > 150):
        st.error("Glucose Level should be a valid number between 70 and 126 mg/dL.")
    
    if BloodPressure and (not BloodPressure.replace('.', '', 1).isdigit() or float(BloodPressure) < 72 or float(BloodPressure) > 130):
        st.error("Blood Pressure Value should be a valid number between 80 and 130 mm Hg.")
    
    if SkinThickness and not SkinThickness.isdigit():
        st.error("Skin Thickness Value should be a valid number.")
    
    if Insulin and (not Insulin.replace('.', '', 1).isdigit() or float(Insulin) < 0 or float(Insulin) > 25):
        st.error("Insulin Level should be a valid number between 2 and 25 µU/mL.")
    
    if BMI and (not BMI.replace('.', '', 1).isdigit() or float(BMI) < 18 or float(BMI) > 35):
        st.error("BMI Value should be a valid number between 18 and 30.")
    
    if DiabetesPedigreeFunction and (not DiabetesPedigreeFunction.replace('.', '', 1).isdigit() or float(DiabetesPedigreeFunction) < 0.08 or float(DiabetesPedigreeFunction) > 2.42):
        st.error("Diabetes Pedigree Function Value should be a valid number between 0.08 and 2.42.")
    
    if Age and (not Age.isdigit() or int(Age) < 0 or int(Age) > 150):
        st.error("Age of the Person should be a valid number between 0 and 150.")

    # Code for Prediction
    diabetes_diagnosis = " "
    
    # Creating a button for Prediction
    if st.button("Diabetes Test Result"):
        try:
            # Convert inputs to numeric values
            Pregnancies = int(Pregnancies)
            Glucose = int(Glucose)
            BloodPressure = int(BloodPressure)
            SkinThickness = int(SkinThickness)
            Insulin = int(Insulin)
            BMI = float(BMI)
            DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
            Age = int(Age)

            diabetes_prediction = diabetes.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

            if diabetes_prediction[0] == 0:
                diabetes_diagnosis = "Hurrah! You have no Diabetes."
            else:
                diabetes_diagnosis = "Sorry! You have Diabetes."
        except Exception as e:
            diabetes_diagnosis = str(e)

    st.success(diabetes_diagnosis)

    # st.subheader("Note: ")
    
    # st.markdown("1. **Number of Pregnancies**") 
    # st.write(" There isn't a normal or abnormal range for the number of pregnancies. It's a count and depends on an individual's medical history.")

    # st.markdown("2. **Glucose Level**:")
    # st.write("- **Normal Range:** Fasting blood sugar levels between 70-99 mg/dL.")
    # st.write("- **Abnormal Range:** Levels above 126 mg/dL on two separate tests may indicate diabetes.")

    # st.markdown("3. **Blood Pressure Value**:")
    # st.write("- **Normal Range:** Typically around 120/80 mm Hg.")
    # st.write("- **Abnormal Range:** High blood pressure is diagnosed when consistently above 130/80 mm Hg.")

    # st.markdown("4. **Skin Thickness Value**:")
    # st.write("Skin thickness is not a standard parameter for diabetes prediction. It might be used for other medical assessments, and there's no specific normal or abnormal range.")

    # st.markdown("5. **Insulin Level**:")
    # st.write("- **Normal Range:** Fasting insulin levels between 2 to 25 µU/mL.")
    # st.write("- **Abnormal Range:** Deviation from this range might indicate issues, but interpretation depends on various factors.")

    # st.markdown("6. **BMI Value (Body Mass Index)**:")
    # st.write("- **Normal Range:** BMI between 18.5 to 24.9.")
    # st.write("- **Overweight:** BMI between 25 to 29.9.")
    # st.write("- **Obesity:** BMI of 30 or greater.")

    # st.markdown("7. **Diabetes Pedigree Function Value**:")
    # st.write("This is a unitless score, and there isn't a specific normal or abnormal range. Higher values generally indicate a higher risk.")

    # st.markdown("8. **Age of the Person**:")
    # st.write("There isn't a normal or abnormal range for age in the context of diabetes prediction. However, diabetes risk generally increases with age.")




#Heart Disease Prediction Page:

if(selected == "❤️ Heart Disease Prediction"):
    #page title
    st.title("Heart Disease Prediction using Machine Learning")
    
# getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age', placeholder="Enter age")
        
    with col2:
        sex = st.text_input('Sex', placeholder="Enter sex (0 or 1)")
    
    with col3:
        cp = st.text_input('Chest Pain types (0, 1, 2, or 3)', placeholder="Enter chest pain type")
    
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', placeholder="60 - 200")
    
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', placeholder="120 - 560 mg/dL")
    
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl', placeholder="0 or 1")
    
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results', placeholder="0 or 1")
    
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved', placeholder="60 - 220 bpm")
    
    with col3:
        exang = st.text_input('Exercise Induced Angina', placeholder="0 or 1")
    
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise', placeholder="0.0 - 6.2")
    
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment', placeholder="0, 1, or 2")
    
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy', placeholder="0 to 4")
    
    with col1:
        thal = st.text_input('thal: 1 = normal; 2 = fixed defect; 3 = reversible defect', placeholder="Enter thal (0 to 3)")
          

    # Validation
    if age and not age.isdigit():
        st.error("Age should be a valid number.")
    elif age and (int(age) < 0 or int(age) > 150):
        st.error("Age should be between 0 and 150.")
    
    if sex and sex not in ('0', '1'):
        st.error("Sex should be either 0 or 1.")
    
    if cp and cp not in ('0', '1', '2', '3'):
        st.error("Chest pain type should be 0, 1, 2, or 3.")
    
    if trestbps and (not trestbps.replace('.', '', 1).isdigit() or float(trestbps) < 60 or float(trestbps) > 200):
        st.error("Resting blood pressure should be a valid number between 60 and 200.")
    
    if chol and (not chol.replace('.', '', 1).isdigit() or float(chol) < 120 or float(chol) > 560):
        st.error("Serum cholesterol should be a valid number between 120 and 560.")
    
    if fbs and fbs not in ('0', '1'):
        st.error("Fasting blood sugar should be either 0 or 1.")
    
    if restecg and restecg not in ('0', '1'):
        st.error("Resting electrocardiographic results should be either 0 or 1.")
    
    if thalach and (not thalach.replace('.', '', 1).isdigit() or float(thalach) < 60 or float(thalach) > 220):
        st.error("Maximum heart rate achieved should be a valid number between 60 and 220.")
    
    if exang and exang not in ('0', '1'):
        st.error("Exercise induced angina should be either 0 or 1.")
    
    if oldpeak and (not oldpeak.replace('.', '', 1).isdigit() or float(oldpeak) < 0.0 or float(oldpeak) > 6.2):
        st.error("ST depression induced by exercise should be a valid number between 0.0 and 6.2.")
    
    if slope and slope not in ('0', '1', '2'):
        st.error("Slope of the peak exercise ST segment should be 0, 1, or 2.")
    
    if ca and (not ca.isdigit() or int(ca) < 0 or int(ca) > 4):
        st.error("Major vessels colored by fluoroscopy should be a valid number between 0 and 4.")
    
    if thal and thal not in ('0', '1', '2', '3'):
        st.error("Thal should be 0, 1, 2, or 3.")
     
    # code for Prediction
    heart_diagnosis = " "
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
            age = float(age)
            sex = int(sex)
            cp = int(cp)
            trestbps = float(trestbps)
            chol = float(chol)
            fbs = int(fbs)
            restecg = int(restecg)
            thalach = float(thalach)
            exang = int(exang)
            oldpeak = float(oldpeak)
            slope = int(slope)
            ca = float(ca)
            thal = float(thal)
            heart_prediction = heart_disease.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])

            if (heart_prediction[0] == 0):
                heart_diagnosis = "Hurrah! Your Heart is Good."
            else:
                heart_diagnosis = "Sorry! You have Heart Problem."
        
    st.success(heart_diagnosis)

    

    # st.subheader("Note :")

    # st.markdown("1. **CP (Chest Pain Type):**")
    # st.write("- **Minimum:** 0.")
    # st.write("- **Maximum:** 3.")

    # st.markdown("2. **Trestbps (Resting Blood Pressure):**")
    # st.write("- **Minimum:** Around 90 mm Hg.")
    # st.write("- **Maximum:** Around 200 mm Hg.")

    # st.markdown("3. **Chol (Serum Cholesterol):**")
    # st.write("- **Minimum:** Around 100 mg/dL.")
    # st.write("- **Maximum:** Around 600 mg/dL.")

    # st.markdown("4. **Fbs (Fasting Blood Sugar):**")
    # st.write("- **Minimum:** Around 70 mg/dL.")
    # st.write("- **Maximum:** Around 130 mg/dL.")

    # st.markdown("5. **Restecg (Resting Electrocardiographic Results):**")
    # st.write("- **Minimum:** 0.")
    # st.write("- **Maximum:** 2.")

    # st.markdown("6. **Thalach (Maximum Heart Rate Achieved):**")
    # st.write("- **Minimum:** Around 60 bpm.")
    # st.write("- **Maximum:** Around 220 bpm.")

    # st.markdown("7. **Exang (Exercise Induced Angina):**")
    # st.write("- **Minimum:** 0 (no angina).")
    # st.write("- **Maximum:** 1 (angina induced during exercise).")

    # st.markdown("8. **Oldpeak (ST Depression Induced by Exercise Relative to Rest):**")
    # st.write("- **Minimum:** Usually starts from 0.")
    # st.write("- **Maximum:** Can vary, often below 6.")

    # st.markdown("9. **Slope:**")
    # st.write("- **Minimum:** 0.")
    # st.write("- **Maximum:** 2.")

    # st.markdown("10. **Ca (Number of Major Vessels Colored by Fluoroscopy):**")
    # st.write("- **Minimum:** 0 (no major vessels colored).")
    # st.write("- **Maximum:** Typically 3 or 4 (varies based on the dataset coding).")

    # st.markdown("11. **Thal (Thalassemia):**")
    # st.write("- **Minimum:** 0 .")
    # st.write("- **Maximum:** 3 .")

        
#Parkinsons Disease Prediction Page:
if(selected == "🧠 Parkinsons Disease Prediction"):
    #page title
    st.title("Parkinsons Disease Prediction using Machine Learning")

# getting the input data from the user
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input("MDVP Fo(Hz)")
        
    with col2:
        fhi = st.text_input("MDVP Fhi(Hz)")
        
    with col3:
        flo = st.text_input("MDVP Flo(Hz)")
        
    with col4:
        Jitter_percent = st.text_input("MDVP Jitter(%)")
        
    with col5:
        Jitter_Abs = st.text_input("MDVP Jitter(Abs)")
        
    with col1:
        RAP = st.text_input("MDVP RAP")
        
    with col2:
        PPQ = st.text_input("MDVP PPQ")
        
    with col3:
        DDP = st.text_input("Jitter DDP")
        
    with col4:
        Shimmer = st.text_input("MDVP Shimmer")
        
    with col5:
        Shimmer_dB = st.text_input("MDVP Shimmer(dB)")
        
    with col1:
        APQ3 = st.text_input("Shimmer APQ3")
        
    with col2:
        APQ5 = st.text_input("Shimmer APQ5")
        
    with col3:
        APQ = st.text_input("MDVP APQ")
        
    with col4:
        DDA = st.text_input("Shimmer DDA")
        
    with col5:
        NHR = st.text_input("NHR")
        
    with col1:
        HNR = st.text_input("HNR")
        
    with col2:
        RPDE = st.text_input("RPDE")
        
    with col3:
        DFA = st.text_input("DFA")
        
    with col4:
        spread1 = st.text_input("spread1")
        
    with col5:
        spread2 = st.text_input("spread2")
        
    with col1:
        D2 = st.text_input("D2")
        
    with col2:
        PPE = st.text_input("PPE")
        
    # code for Prediction
    parkinsons_diagnosis = " "
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
            parkinsons_prediction = parkinsons_disease.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
            if (parkinsons_prediction[0] == 0):
                parkinsons_diagnosis = "Hurrah! You don't have Parkinson's Disease."
            else:
                parkinsons_diagnosis = "Sorry! You have Parkinson's Disease."
        
    st.success(parkinsons_diagnosis)  



# Breast Cancer Prediction Page:
if selected == "🎗️ Breast Cancer Prediction":

    # Page title
    st.title("Breast Cancer Prediction using Machine Learning")

    # Getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        # Ensure you define the input variable names correctly
        # Use 'uploaded_image' for the uploaded image
        uploaded_image = st.file_uploader("Please upload the mammograph of Person", type=["jpg", "png", "jpeg"])

    with col2:
        # Optional: Display the uploaded image
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded mammography Image", use_column_width=True)

    # Create a button for prediction
    if st.button("Predict Breast Cancer"):
        pneumonia_diagnosis = " "
        # pneumonia_value = " "

        if uploaded_image is not None:
            # Preprocess the uploaded image and convert it to grayscale
            img = Image.open(uploaded_image)
            img = img.convert('RGB')
            img = img.resize((150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            # Preprocess the image (optional: normalize pixel values)
            img_array = img_array / 255.0

            # Load the pre-trained model
            model = load_model("models/breastcancer.h5")

            # Make a prediction
            pneumonia_prediction = model.predict(img_array)

            # Define your logic to interpret the prediction
            # For example, 0 for uninfected and 1 for infected
            # You should update this part based on your model output
            if pneumonia_prediction > 0.5:
                pneumonia_diagnosis = "You have Breast Cancer. Please consult a Doctor."
            else:
                pneumonia_diagnosis = "You are healthy!"

        st.success(pneumonia_diagnosis)
        # st.success(pneumonia_value)



#Lung Cancer Prediction Page:

if(selected == "🫁 Lung Cancer Prediction"):
    
    #page title
    st.title("Lung Cancer Prediction using Machine Learning")
    st.markdown("Note: Enter 1 for No and 2 for Yes")

# getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        GENDER = st.number_input("Gender (1 = M 0 = F)" , min_value=0, max_value=1, step=1)
        
    with col2:
        AGE = st.number_input("Age", min_value=0)
    
    with col3:
        SMOKING = st.number_input("Smoking", min_value=1, max_value=2, step=1)
    
    with col4:
        YELLOW_FINGERS = st.number_input("Yellow Fingers", min_value=1, max_value=2, step=1)
    
    with col1:
        ANXIETY = st.number_input("Anxiety", min_value=1, max_value=2, step=1)
    
    with col2:
        PEER_PRESSURE = st.number_input("Peer Pressure", min_value=1, max_value=2, step=1)
    
    with col3:
        CHRONIC_DISEASE = st.number_input("Chronic Disease", min_value=1, max_value=2, step=1)
    
    with col4:
        FATIGUE = st.number_input("Fatigue", min_value=1, max_value=2, step=1)
    
    with col1:
        ALLERGY = st.number_input("Allergy", min_value=1, max_value=2, step=1)
    
    with col2:
        WHEEZING = st.number_input("Wheezing", min_value=1, max_value=2, step=1)
    
    with col3:
        ALCOHOL_CONSUMING = st.number_input("Alcohol Consuming", min_value=1, max_value=2, step=1)
    
    with col4:
        COUGHING = st.number_input("Coughing", min_value=1, max_value=2, step=1)
    
    with col1:
        SHORTNESS_OF_BREATH = st.number_input("Shortness Of Breath", min_value=1, max_value=2, step=1)
    
    with col2:
        SWALLOWING_DIFFICULTY = st.number_input("Swallowing Difficulty", min_value=1, max_value=2, step=1)
    
    with col3:
        CHEST_PAIN = st.number_input("Chest Pain", min_value=1, max_value=2, step=1)
    


# code for Prediction
    lung_cancer_result = " "
    
    # creating a button for Prediction
    
    if st.button("Lung Cancer Test Result"):
        lung_cancer_report = lung_cancer.predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
        
        if (lung_cancer_report[0] == 0):
          lung_cancer_result = "Hurrah! You have no Lung Cancer."
        else:
          lung_cancer_result = "Sorry! You have Lung Cancer."
        
    st.success(lung_cancer_result)


# Malaria Prediction Page:
if selected == "🦟 Malaria Prediction":
    # Page title
    st.title("Malaria Prediction using Machine Learning")

    # Getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        # Ensure you define the input variable names correctly
        # Use 'uploaded_image' for the uploaded image
        uploaded_image = st.file_uploader("Upload an image of the cell", type=["jpg", "png", "jpeg"])

    with col2:
        # Optional: Display the uploaded image
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Create a button for prediction
    if st.button("Predict Malaria"):
        malaria_diagnosis = " "

        if uploaded_image is not None:
            try:
                # Preprocess the uploaded image
                img = Image.open(uploaded_image)
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)

                # Load the pre-trained model
                model = load_model("models/malaria_new.h5")

                # Make a prediction
                malaria_prediction = np.argmax(model.predict(img)[0])

                # Define your logic to interpret the prediction
                # For example, 0 for uninfected and 1 for infected
                # You should update this part based on your model output
                if malaria_prediction < 0.5:
                    malaria_diagnosis = "This cell is not Infected."
                else:
                    malaria_diagnosis = "This cell is an Infected Malaria Cell."

            except Exception as e:
                malaria_diagnosis = "An error occurred. Please make sure to upload a valid image."

        st.success(malaria_diagnosis)


# Pneumonia Prediction Page:
if selected == "🤧 Pneumonia Prediction":
    # Page title
    st.title("Pneumonia Prediction using Machine Learning")

    # Getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        # Ensure you define the input variable names correctly
        # Use 'uploaded_image' for the uploaded image
        uploaded_image = st.file_uploader("Please upload the X-Ray of Person", type=["jpg", "png", "jpeg"])

    with col2:
        # Optional: Display the uploaded image
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded X-Ray Image", use_column_width=True)

    # Create a button for prediction
    if st.button("Predict Pneumonia"):
        pneumonia_diagnosis = " "

        if uploaded_image is not None:
            try:
                # Preprocess the uploaded image and convert it to grayscale
                img = Image.open(uploaded_image).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))
                img = img / 255.0

                # Load the pre-trained model
                model = load_model("models/pneumonia.h5")

                # Make a prediction
                pneumonia_prediction = np.argmax(model.predict(img)[0])

                # Define your logic to interpret the prediction
                # For example, 0 for uninfected and 1 for infected
                # You should update this part based on your model output
                if pneumonia_prediction > 0.5:
                    pneumonia_diagnosis = "You have pneumonia. Please consult a Doctor."
                else:
                    pneumonia_diagnosis = "You are healthy!"

            except Exception as e:
                pneumonia_diagnosis = "An error occurred. Please make sure to upload a valid image."

        st.success(pneumonia_diagnosis)