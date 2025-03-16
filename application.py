import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline


st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
st.title(" Don Concentration Prediction App")
st.write("Upload a CSV file with **448 features** to predict crop yield.")
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    
    input_data = pd.read_csv(uploaded_file)

    
    if input_data.shape[1] != 448:
        st.error(f"❌ Expected 448 features, but got {input_data.shape[1]}. Please check your file.")
    else:
        st.success("✅ File successfully uploaded and verified!")
        pipeline = PredictPipeline()

        
        predictions = pipeline.predict(input_data)

        
        st.subheader("📊 Predicted Crop Yield Values:")
        st.write(pd.DataFrame(predictions, columns=["Predicted Yield"]))

        
        output_df = pd.DataFrame(predictions, columns=["Predicted Yield"])
        csv_output = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Predictions", data=csv_output, file_name="predictions.csv", mime="text/csv")


