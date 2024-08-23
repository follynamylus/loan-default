import streamlit as st
import pandas as pd 
import pickle as pkl
from sklearn.preprocessing import StandardScaler

st.title("Credit Risk Prediction Using Xtreme Gradient Boosting Algorithm (XGBoost)")


st.sidebar.title("Input The Feature Values")
pa = st.sidebar.number_input(label= "person_age", min_value= 18, max_value= 90)
pi = st.sidebar.number_input(label= "person_income", min_value= 2000, max_value= 6000000)
pel = st.sidebar.number_input(label= "person_emp_length", min_value= 0.00, max_value= 100.00)
la = st.sidebar.number_input(label= "loan_amnt", min_value= 500, max_value= 50000)
lir = st.sidebar.number_input(label= "loan_int_rate", min_value= 5.0, max_value= 25.0)
lpi = st.sidebar.number_input(label= "loan_percent_income", min_value= 0.00, max_value= 0.85)
cbp = st.sidebar.number_input(label= "cb_person_cred_hist_length", min_value= 2, max_value= 35)

pho = st.sidebar.selectbox( label='person_home_ownership', options=["RENT","MORTGAGE","OWN","OTHER"])
li = st.sidebar.selectbox(label='loan_intent', options=["EDUCATION","MEDICAL","VENTURE",
                                                        "PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"])
lg = st.sidebar.selectbox(label='loan_grade', options=["A","B","C","D","E","F","G"])
cbdf = st.sidebar.selectbox(label='cb_person_default_on_file', options=["Y","N"])


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
df['Loan_to_Income_Ratio'] = df['loan_amnt'] / df['person_income']
df['Age_Group'] = pd.cut(df['person_age'], bins=[0, 25, 40, 65, 90], labels=['Young', 'Adult', 'Senior', 'Elderly'])

cats = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file",'Age_Group']
en_df = pd.get_dummies(df[cats])
df.drop(cats, axis = 1, inplace=True)
df = df.join(en_df)
dt = {"person_age":[False],"person_income":[False],"person_emp_length":[False],"loan_amnt":[False],"loan_int_rate":[False],
        "loan_percent_income":[False],
        "cb_person_cred_hist_length":[False],"Loan_to_Income_Ratio":[False],
        "person_home_ownership_MORTGAGE":[False],"person_home_ownership_OTHER":[False],
        "person_home_ownership_OWN":[False],"person_home_ownership_RENT":[False],"loan_intent_DEBTCONSOLIDATION":[False],
        "loan_intent_EDUCATION":[False],
        "loan_intent_HOMEIMPROVEMENT":[False],'loan_intent_MEDICAL':[False],'loan_intent_PERSONAL':[False],"loan_intent_VENTURE":[False],
        "loan_grade_A":[False],'loan_grade_B':[False],'loan_grade_C':[False],'loan_grade_D':[False],'loan_grade_E':[False],'loan_grade_F':[False],
        "loan_grade_G":[False],'cb_person_default_on_file_N':[False],'cb_person_default_on_file_Y':[False],
        "Age_Group_Young":[False],"Age_Group_Adult":[False],"Age_Group_Senior":[False],
        "Age_Group_Elderly":[False]}
dt_df = pd.DataFrame(dt)
for i in dt_df.columns :
    if i in df.columns :
        dt_df[i] = df[i]
    else :
        dt_df[i] = dt_df[i]
scaler = StandardScaler()
scaled_df = scaler.fit_transform(dt_df)
with open('xgboost.pkl', 'rb') as f:
    rf_model = pkl.load(f)
    pred = rf_model.predict(scaled_df)
if pred[0] == 0 :
    prediction = "Not Default"
else : 
    prediction = "Default"

if st.button("Click for Prediction"):
    st.success("Prediction")
    st.text(f"The customer is predicted to {prediction}")