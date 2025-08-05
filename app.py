import streamlit as st
import pandas as pd
import openml 
import numpy as np
import joblib


st.set_page_config(
    page_title = "Loan Credit Aprroval App",
    layout = 'wide' 
)

openml.config.apikey = "5e0e46f6f2efce9de5af4d77313dab50"
dataset = openml.datasets.get_dataset(46365)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)


df = X.copy()
df['age_group'] = pd.cut(
    df['age'],
    bins = [20,30,40,50,60],
    labels = ['20-30', '30-40', '40-50', '50-60']
    
)

df['credit_amount'] = df['credit_amount'].astype(float)


approval_rate = y.astype('category').cat.codes.mean()

avg_credit_ammount = df['credit_amount'].mean()

avg_installment_commitment = df['installment_commitment'].mean()


with st.sidebar:
    
    st.header("Applicant details")
    
    duration               = st.number_input("Duration (months)", 1, 100, 50)
    credit_amount_input    = st.number_input("Credit Amount", min_value=0.0, value=1000.0)
    installment_commitment = st.selectbox("Installment Commitment", ['Very-High', 'High', 'Medium', 'Low'])
    residence_since        = st.selectbox("Residence Since (years)", ['>4','3','2','1'])
    age_input              = st.number_input("Age (years)", 18, 100, 30)
    credit_history         = st.selectbox("Credit History", ['All Paid','existing credit','delayed previously','existing paid','no credits/all paid'])
    job                     = st.selectbox("Job", ['High Qualified/self employed/mgmt','skilled','unskilled/unemployed -non resident','unskilled/unemployed - resident'])
    employment_period      = st.selectbox("Employment Period", ['More than 7 years','Between 7 and 4 years','Between 4 and 1 years','None'])
    purpose                = st.selectbox("Purpose", ['Business','domestic appliance','education','furniture/equipment','new car','other','radio/tv','repairings','restraining','used car'])
    personal_status        = st.selectbox("Personal Status", ['Female div/dep/mar','male div/sep','male mar/wid','male single'])
    property_magnitude     = st.selectbox("Property Magnitude", ['Car','Life Insurance','No Known Property','Real Estate'])
    housing                = st.selectbox("Housing", ['Own','For Free','Rent'])
    foreign_worker         = st.radio("Foreign Worker", ['Yes','No'])
    own_telephone          = st.radio("Own Telephone", ['None','Yes'])
    other_parties          = st.selectbox("Other Parties", ['co-applicant','guarantor','None'])
    other_payment_plans    = st.selectbox("Other Payment Plans", ['bank','None','stores'])
    savings_status         = st.selectbox("Saving Status", ['>1000','500<=,<1000','100<=,<500','<100','No Known Savings'])
    existing_credits       = st.selectbox("Existing Credit", [1,2,3,4])
    num_dependents         = st.selectbox("Number of Dependents", [1,2])
    predict_button         = st.button("Predict")


st.title("Loan Credit Approval")


col1, col2, col3 = st.columns(3)

col1.metric("Approval Rate", f"{approval_rate:.1%}")
col2.metric("Avg Credit Amount Issued by Bank", f"{avg_credit_ammount:,.4f}$")
col3.metric("Average installment commitment:", f'{avg_installment_commitment:.0f}')

col_chart1,  col_chart2 = st.columns((2,1))

with col_chart1:
    
    st.subheader("Installment_commitment by age")
    st.bar_chart(X.groupby('age')['installment_commitment'].sum())
with col_chart2:
    
    st.subheader('Credit amount issued by Age')
    st.bar_chart(df.groupby('age_group')['credit_amount'].sum())
    
    
with st.expander("Show Raw Data"):
    
    
    
    st.dataframe(X.head(), use_container_width=True)
    
    
if predict_button:
    # load the trained model
    model = joblib.load("credit_approval_model.pkl")

    # mapping for categorical inputs
    mappings = {
        "installment_commitment": {'Very-High':4,'High':3,'Medium':2,'Low':1},
        "residence_since":        {'>4':0,'3':1,'2':2,'1':3},
        "credit_history":         {'All Paid':0,'existing credit':1,'delayed previously':2,'existing paid':3,'no credits/all paid':4},
        "job":                    {'High Qualified/self employed/mgmt':0,'skilled':1,'unskilled/unemployed -non resident':2,'unskilled/unemployed - resident':3},
        "employment_period":      {'More than 7 years':0,'Between 7 and 4 years':1,'Between 4 and 1 years':2,'None':3},
        "purpose":                {'Business':0,'domestic appliance':1,'education':2,'furniture/equipment':3,'new car':4,'other':5,'radio/tv':6,'repairings':7,'restraining':8,'used car':9},
        "personal_status":        {'Female div/dep/mar':0,'male div/sep':1,'male mar/wid':2,'male single':3},
        "property_magnitude":     {'Car':0,'Life Insurance':1,'No Known Property':2,'Real Estate':3},
        "housing":                {'Own':0,'For Free':1,'Rent':2},
        "foreign_worker":         {'Yes':1,'No':0},
        "own_telephone":          {'None':0,'Yes':1},
        "other_parties":          {'co-applicant':0,'guarantor':1,'None':2},
        "other_payment_plans":    {'bank':0,'None':1,'stores':2},
        "saving_status":          {'>1000':3,'500<=,<1000':0,'100<=,<500':1,'<100':4,'No Known Savings':2},
    }

    # prepare the feature vector
    input_data = np.array([
        duration,
        credit_amount_input,
        mappings["installment_commitment"][installment_commitment],
        mappings["residence_since"][residence_since],
        age_input,
        existing_credits,
        num_dependents,
        mappings["saving_status"][savings_status],
        mappings["credit_history"][credit_history],
        mappings["job"][job],
        mappings["employment_period"][employment_period],
        mappings["purpose"][purpose],
        mappings["personal_status"][personal_status],
        mappings["property_magnitude"][property_magnitude],
        mappings["housing"][housing],
        mappings["own_telephone"][own_telephone],
        mappings["foreign_worker"][foreign_worker],
        mappings["other_parties"][other_parties],
        mappings["other_payment_plans"][other_payment_plans],
    ]).reshape(1, -1)


    pred = model.predict(input_data)[0]

    proba = model.predict_proba(input_data)[0,1]

    st.subheader("Probability distribuion graph")

    rcol1, rcol2 = st.columns(2)

    rcol1.metric("Decision", "Approved" if pred == 1 else "Denied")
    rcol2.metric("Confidence", f"{proba:.1%}")

