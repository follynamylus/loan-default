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

tab1, tab2, tab3 = st.tabs(["Data Input","Predictions","About the App"])

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

    if tab2.button("Click for Prediction"):
        tab2.success("Prediction")
        tab2.text(f"The customer is predicted to {prediction}")
        tab2.write(f"The probability of the credit default is {pred_proba[:,1] * 100} %")

        fig = sns.barplot(x=np.arange(len(pred_proba[0])), y=pred_proba[0])
        plt.xticks(np.arange(len(pred_proba[0])), labels=[f"Class {i}" for i in range(len(pred_proba[0]))])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title(f'Predicted Probabilities for Random Forest')
        tab2.pyplot()

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
        
        tab2.dataframe(dp)

        tab2.success("Credit Risk Prediction Bar Plot")
        fig = sns.countplot(data = dp, x= "Prediction")
        plt.savefig('predicted_values.png')
        tab2.pyplot()