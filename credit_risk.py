import streamlit as st
import pandas as pd 
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)


st.sidebar.title("Select type of prediction")
inst = st.sidebar.selectbox("Select Prefered Prediction", options=["Single Instance", "Multiple Instances"])

with open('model.pkl', 'rb') as f:
    rf_model = pkl.load(f)

tab3, tab1, tab2 = st.tabs(["ABOUT THE APP","DATA INPUT","PREDICTION"])

if inst.lower() == "single instance" :

    with tab1 :

        tab1.subheader("Input feature values")
        col1, col2 = tab1.columns(2)

        with col1 :
            pa = col1.number_input(label= "person_age", min_value= 18, max_value= 90)
            pi = col1.number_input(label= "person_income", min_value= 2)
            pel = col1.number_input(label= "person_emp_length", min_value= 0.00, max_value= 100.00)
            la = col1.number_input(label= "loan_amnt", min_value= 5)
            lir = col1.number_input(label= "loan_int_rate", min_value= 5.0, max_value= 25.0)

        with col2 :
            lpi = col2.number_input(label= "loan_percent_income", min_value= 0.00, max_value= 0.85)
            cbp = col2.number_input(label= "cb_person_cred_hist_length", min_value= 2, max_value= 35)
            pho = col2.selectbox( label='person_home_ownership', options=["RENT","MORTGAGE","OWN","OTHER"])
            li = col2.selectbox(label='loan_intent', options=["EDUCATION","MEDICAL","VENTURE",
                                                                    "PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"])
            lg = col2.selectbox(label='loan_grade', options=["A","B","C","D","E","F","G"])
            cbdf = col2.selectbox(label='cb_person_default_on_file', options=["Y","N"])
        

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

    cats = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]
    en_df = pd.get_dummies(df[cats])
    df.drop(cats, axis = 1, inplace=True)
    df = df.join(en_df)
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
    dt_df = pd.DataFrame(dt)
    for i in dt_df.columns :
        if i in df.columns :
            dt_df[i] = df[i]
        else :
            dt_df[i] = dt_df[i]
        pred = rf_model.predict(dt_df)
        pred_proba = rf_model.predict_proba(dt_df)
    if pred[0] == 0 :
        prediction = "Not Default"
    else : 
        prediction = "Default"

    tab2.success("Credit Risk Prediction Using Extreme Gradient Boosting Classifier Algorithm")
    if tab2.button("Click for Prediction"):
        tab2.success("Prediction")
        tab2.text(f"The customer is predicted to {prediction}")
        tab2.write(f"The probability of the credit default is {pred_proba[:,1] * 100} %")

        fig = plt.figure(figsize=(2,2))
        sns.barplot(x=np.arange(len(pred_proba[0])), y=pred_proba[0], color= 'brown')
        plt.xticks(np.arange(len(pred_proba[0])), labels=[f"Default {i}" for i in range(len(pred_proba[0]))])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title(f'Probability for default')
        tab2.pyplot(fig,use_container_width=False)

elif inst.lower() == "multiple instances" :
    mult = tab1.file_uploader("Upload a CSV file ")
    if mult == None :
        tab1.write("Load in a CSV file")
    else :
        df = pd.read_csv(mult)
        dp = df.copy()
        cats = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]
        en_df = pd.get_dummies(df[cats])
        df.drop(cats, axis = 1, inplace=True)
        df = df.join(en_df)
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
        dt_df = pd.DataFrame(dt,index=(range(len(df))))
        for i in dt_df.columns :
            if i in df.columns :
                dt_df[i] = df.loc[:,i]
            else :
                dt_df[i] = dt_df[i]
                
    
        dt_df = dt_df.fillna(0)

        pred = rf_model.predict(dt_df)
        pred_proba = rf_model.predict_proba(dt_df)[:,1] * 100

        predict = []
        for i in range(len(pred)) :
            if pred[i] == 0 :
                predict.append("Not Default")
            else : 
                predict.append("Default")

        dp["Prediction"] = predict
        dp["Default Probability %"] = pred_proba
        
        tab2.success("Credit Risk Prediction Using Extreme Gradient Boosting Classifier Algorithm")
        tab2.dataframe(dp)

        
        tab2.success("Credit Risk Prediction Bar Plot")
        fig = plt.figure(figsize=(2,2))
        sns.countplot(data = dp, x= "Prediction", color= "red")
        tab2.pyplot(fig,use_container_width=False)

tab3.success("Credit Risk Prediction Using Extreme Gradient Boosting Classifier Algorithm")
tab3.write(
    """
    The Web Application is used for making sales forecasts / predictions of a fast food enterprise.

    The web application makes use of statistical models trained using a data modeled on Item-7 sales from 1st of January 2015 till 31st if December 2022.
    The data which are aggregated to the average monthly value for the weather features.

    The application consists of an adjustable Sidebar for Input widgets,
    Visualizations are provided on the first tab while data frame of the predictions and download option are provided on the second tabs of the application
    """
)

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
#tab3.subheader("ABOUT OUTPUT WIDGETS")
#tab3.write(
  #  """
   # The Output widgets are in tabs .

   # Tab 1  the PREDICTION AND DOWNLOAD tab, It displays the prediction dataframe and a download button option for downloading the file in a 
   # CSV format.

   # Tab 2 the VISUALIZATION tab contains graphs and plots of the predictions. The graphs are in expanders that include the Line plot , Bar chart ,
   # stacked Area plot and stacked Density Contour plot . These variables are flexible and multiple can be selected at a time,
    # it also can be activated or deactivated by clicking on their names by the top
    #right corner of Visualization tab, all these are for multiple predictions.

    #Tab 2 for single prediction displays values for the date and the date the sales were made.


    #Tab 3 ABOUT APPLICATION tab, contains information on the application widgets .
    #"""
#)