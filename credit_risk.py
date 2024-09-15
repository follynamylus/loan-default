## Import the required packages

import streamlit as st
import pandas as pd 
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)

## Set the dropdown for prediction choice.
st.sidebar.title("Select type of prediction")
inst = st.sidebar.selectbox("Select Prefered Prediction", options=["Single Instance", "Multiple Instances"])

## Load in the model 
with open('model.pkl', 'rb') as f:
    xgb_model = pkl.load(f)

## Create the tabs
tab3, tab1, tab2 = st.tabs(["ABOUT THE APP","DATA INPUT","PREDICTION"])

## Condition code for single instance prediction
if inst.lower() == "single instance" :

## Operations to perform in the first tab for single instance prediction
    with tab1 :

    ## Declare the header for first tab of single prediction 
        tab1.subheader("Input feature values")

    ## Divide the tab into two columns
        col1, col2 = tab1.columns(2)
    ## Create input widgets for first column
        with col1 :
            pa = col1.number_input(label= "person_age", min_value= 18, max_value= 90)
            pi = col1.number_input(label= "person_income", min_value= 2)
            pel = col1.number_input(label= "person_emp_length", min_value= 0.00, max_value= 100.00)
            la = col1.number_input(label= "loan_amnt", min_value= 5)
            lir = col1.number_input(label= "loan_int_rate", min_value= 5.0, max_value= 25.0)
    ## Create Input widgets for second column
        with col2 :
            lpi = col2.number_input(label= "loan_percent_income", min_value= 0.00, max_value= 0.85)
            cbp = col2.number_input(label= "cb_person_cred_hist_length", min_value= 2, max_value= 35)
            pho = col2.selectbox( label='person_home_ownership', options=["RENT","MORTGAGE","OWN","OTHER"])
            li = col2.selectbox(label='loan_intent', options=["EDUCATION","MEDICAL","VENTURE",
                                                                    "PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"])
            lg = col2.selectbox(label='loan_grade', options=["A","B","C","D","E","F","G"])
            cbdf = col2.selectbox(label='cb_person_default_on_file', options=["Y","N"])
        
# Create a dataframe from the input
    df = pd.DataFrame()
    df["person_age"] = [pa]
    df['person_income'] = [pi]
    df['person_emp_length'] = [pel]
    df['loan_amnt'] = [la]
    df['loan_int_rate'] = [lir]
    df['loan_percent_income'] = [lpi]
    df['cb_person_cred_hist_length'] = [cbp]

    df['person_home_ownership'] = [pho]
    df['loan_intent'] = [li]
    df['loan_grade'] = [lg]
    df['cb_person_default_on_file'] = [cbdf]

## Encode categorical columns from the dataframe
    cats = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]
    en_df = pd.get_dummies(df[cats])
    df.drop(cats, axis = 1, inplace=True)
    df = df.join(en_df)
## Create a placeholder dataframe that contains all features needed by the loaded model
    dt = {"person_age":[False],"person_income":[False],"person_emp_length":[False],"loan_amnt":[False],"loan_int_rate":[False],
            "loan_percent_income":[False],
            "cb_person_cred_hist_length":[False],
            "person_home_ownership_MORTGAGE":[False],"person_home_ownership_OTHER":[False],
            "person_home_ownership_OWN":[False],"person_home_ownership_RENT":[False],"loan_intent_DEBTCONSOLIDATION":[False],
            "loan_intent_EDUCATION":[False],
            "loan_intent_HOMEIMPROVEMENT":[False],'loan_intent_MEDICAL':[False],'loan_intent_PERSONAL':[False],"loan_intent_VENTURE":[False],
            "loan_grade_A":[False],'loan_grade_B':[False],'loan_grade_C':[False],'loan_grade_D':[False],'loan_grade_E':[False],'loan_grade_F':[False],
            "loan_grade_G":[False],'cb_person_default_on_file_N':[False],'cb_person_default_on_file_Y':[False]
            }
    ## Compare the placeholder dataframe with the input features dataframe and fill missing features with place holder feature.
    dt_df = pd.DataFrame(dt)
    for i in dt_df.columns :
        if i in df.columns :
            dt_df[i] = df[i]
        else :
            dt_df[i] = dt_df[i]

    # Predict using the resulting dataframe
        pred = xgb_model.predict(dt_df)
        pred_proba = xgb_model.predict_proba(dt_df)
    # Convert the result into Category class.
    if pred[0] == 0 :
        prediction = "Not Default"
    else : 
        prediction = "Default"
    # Create Outputs to display prediction and prediction probability
    tab2.success("Credit Risk Prediction Using Extreme Gradient Boosting Classifier Algorithm")
    if tab2.button("Click for Prediction"):
        tab2.success("Prediction")
        tab2.text(f"The customer is predicted to {prediction}")
        tab2.write(f"The probability of the credit default is {pred_proba[:,1] * 100} %")

## Create a a bar plot for the predict probability.
        fig = plt.figure(figsize=(2,2))
        sns.barplot(x=np.arange(len(pred_proba[0])), y=pred_proba[0], color= 'brown')
        plt.xticks(np.arange(len(pred_proba[0])), labels=[f"Default {i}" for i in range(len(pred_proba[0]))])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title(f'Probability for default')
        tab2.pyplot(fig,use_container_width=False)

# Operations for multiple instances prediction

elif inst.lower() == "multiple instances" :

    ## Create input widget for multiple instances
    mult = tab1.file_uploader("Upload a CSV file ")

    ## Code to handle empty file.
    if mult == None :
        tab1.write("Load in a CSV file")
    
    ## Data processing for multiple instances
    else :
        # Read in the file
        df = pd.read_csv(mult)
        dp = df.copy()
        # Encode categorical features
        cats = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]
        en_df = pd.get_dummies(df[cats])
        df.drop(cats, axis = 1, inplace=True)
        df = df.join(en_df)
        # Create placeholder dataframe
        dt = {"person_age":[False],"person_income":[False],"person_emp_length":[False],"loan_amnt":[False],"loan_int_rate":[False],
                "loan_percent_income":[False],
                "cb_person_cred_hist_length":[False],
                "person_home_ownership_MORTGAGE":[False],"person_home_ownership_OTHER":[False],
                "person_home_ownership_OWN":[False],"person_home_ownership_RENT":[False],"loan_intent_DEBTCONSOLIDATION":[False],
                "loan_intent_EDUCATION":[False],
                "loan_intent_HOMEIMPROVEMENT":[False],'loan_intent_MEDICAL':[False],'loan_intent_PERSONAL':[False],"loan_intent_VENTURE":[False],
                "loan_grade_A":[False],'loan_grade_B':[False],'loan_grade_C':[False],'loan_grade_D':[False],'loan_grade_E':[False],'loan_grade_F':[False],
                "loan_grade_G":[False],'cb_person_default_on_file_N':[False],'cb_person_default_on_file_Y':[False]
                }
        
        # Compare to ensure the features corresponds with what was used to train the model.
        dt_df = pd.DataFrame(dt,index=(range(len(df))))
        for i in dt_df.columns :
            if i in df.columns :
                dt_df[i] = df.loc[:,i]
            else :
                dt_df[i] = dt_df[i]
                
        # Fill missing values with zero
        dt_df = dt_df.fillna(0)

        # Predict and predict probability of multiple features
        pred = xgb_model.predict(dt_df)
        pred_proba = xgb_model.predict_proba(dt_df)[:,1] * 100

        # Convert predictions to categorical classes
        predict = []
        for i in range(len(pred)) :
            if pred[i] == 0 :
                predict.append("Not Default")
            else : 
                predict.append("Default")
        # Include prediction in the dataframe
        dp["Prediction"] = predict
        dp["Default Probability %"] = pred_proba
        
        # Display the prediction dataframe and value counts
        tab2.success("Credit Risk Prediction Using Extreme Gradient Boosting Classifier Algorithm")
        tab2.dataframe(dp)

        tab2.write(dp["Prediction"].value_counts())
         
         # Create columns and plots for multiple prediction
        col1,col2 = tab2.columns(2)
        
        ## Create pie and bar charts in first column
        with col1 :
            fig = px.pie(dp, names= "Prediction", height=300, title= "Prediction Pie chart")
            #fig.update_traces(marker_color='#008060')
            st.plotly_chart(fig, use_container_width=True)
            fig = px.bar(dp, x= "Prediction", y="loan_int_rate", height=300, title= "Prediction against Interest rate")
            fig.update_traces(marker_color='#008060')
            st.plotly_chart(fig, use_container_width=True)

        ## Create Scatter and histogram for the second column
        with col2 :
            fig  = px.scatter(data_frame= dp, x= "Default Probability %", labels= "Prediction",
                              color= "cb_person_default_on_file",
                               title= "prediction probability against Default on file",
                                height= 300 )
            #fig.update_traces(marker_color='#702014')
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(dp, x="Default Probability %", y= "person_income", height=300, title= "Prediction probability against person income")
            fig.update_traces(marker_color='#702014')
            st.plotly_chart(fig, use_container_width=True)

## Create information about the application in first tab.
tab3.success("Credit Risk Prediction Using Extreme Gradient Boosting Classifier Algorithm")

tab3.subheader("ABOUT THE PROJECT")
tab3.write(
    """
It is a classification supervised Machine Learning project, where a loan applicant is predicted to either Default a loan or Not.
This is done through considering some data features of the applicant. Machine Learning technique used in developing the classification system,
relies on the use of past data to train an algorithm to make future prediction. The algorithm was train on 32581 rows of data and 12 Features.

    
"""
)
tab3.subheader("ABOUT THE FEATURES")
tab3.write(
    '''
The features include :
***************************************************************************
1) person_age : A numerical data that represent age of the applicant.

2) person_income : A numerical data that represent the amount the applicant earns annually.

3) person_home_ownership : A categorical feature that depicts the type of home the applicant owns or lives in, it contains classes as RENT, MORTGAGE, OWN, and OTHER

4) person_emp_length : A numerical feature about how long the applicant has been employed in years.

5) loan_intent : A categorical feature that explains the reason for applying for the loan, the classes in it include, EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT.

6) loan_grade : A categorical feature that represent how much the risk for the loan is, that ranges from A - G with A being the least risk and G as the highest risk.

7) loan_amnt : A numerical feature that represent the amount of loan applied for.

8) loan_int_rate : A numerical feature that addresses the rate of interest on the loan.

9) loan_percent_income : A numerical feature that explain the percent of the ratio of loan against income.

10) cb_person_default_on_file : A categorical feature about if the applicant has ever defaulted or not, with classes as Y and N

11) cb_person_cred_hist_length : A numerical feature that represent the amount of credit the applicant has taken in the past.
    
    
    '''
)
tab3.subheader("ABOUT WIDGETS")
tab3.write(
    """
    The Input widgets consist of numerical input type which is in an integer or float format with ranges of values. 

    There is a widget to select the prediction type either single instance or multiple instances.

    The option widget for select box dropdown, for categorical features.
    

    only the prediction category widget is contained in the side bar while the feature input widgets are contained in the second tab of the web application.

    """
)
