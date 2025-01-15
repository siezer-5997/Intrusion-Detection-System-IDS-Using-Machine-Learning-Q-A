import gradio as gr
import openai
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI
import random

# Load the OpenAI API key from the .env file
load_dotenv()
client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the trained models and scaler
log_reg_model = joblib.load('log_reg_model.pkl')  
xgb_model = joblib.load('xgb_model.pkl')  
ada_model = joblib.load('ada_model.pkl') 
gb_model = joblib.load('gb_model.pkl') 
scaler = joblib.load('scaler.pkl') 

# Function to preprocess and predict
def preprocess_and_predict(file):
    # Read the uploaded file
    new_data = pd.read_csv(file.name)

    # Preprocess the data
    new_data.columns = new_data.columns.str.strip()
    new_data_filled = new_data.fillna(0)  # Fill missing values with 0

    # Select the features used for training
    X_new_data = new_data_filled[['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                                   'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s']].values

    # Replace infinite values with 0
    X_new_data = np.where(np.isinf(X_new_data), 0, X_new_data)

    # Scale the input data using the pre-trained scaler
    X_scaled_new_data = scaler.transform(X_new_data)

    # Make predictions using all models
    y_pred_log_reg = log_reg_model.predict(X_scaled_new_data)
    y_pred_xgb = xgb_model.predict(X_scaled_new_data)
    y_pred_ada = ada_model.predict(X_scaled_new_data)
    y_pred_gb = gb_model.predict(X_scaled_new_data)

    # Combine predictions (majority voting)
    combined_predictions = [
        max(set([y_pred_log_reg[i], y_pred_xgb[i], y_pred_ada[i], y_pred_gb[i]]), key=[y_pred_log_reg[i], y_pred_xgb[i], y_pred_ada[i], y_pred_gb[i]].count)
        for i in range(len(y_pred_log_reg))
    ]

    # Map predictions back to readable labels (0 -> 'BENIGN', 1 -> 'DDoS')
    predictions_labels = ['BENIGN' if pred == 0 else 'DDoS' for pred in combined_predictions]

    # Format the predictions into a neat output
    result = pd.DataFrame({
        "Data Point": range(1, len(predictions_labels) + 1),
        "Prediction": predictions_labels
    })

    return result

# Function to simulate DDoS attack data based on patterns in the CSV
def simulate_attack_data():
    attack_types = ['SYN Flood', 'UDP Flood', 'DNS Amplification', 'Ping of Death', 'HTTP Flood']
    
    simulated_data = {
        "Attack Type": random.choices(attack_types, k=5),
        "Packet Count": [random.randint(1000, 10000) for _ in range(5)],
        "Duration (s)": [random.randint(5, 60) for _ in range(5)],
        "Impact Level": random.choices(["Low", "Medium", "High"], k=5)
    }

    simulated_df = pd.DataFrame(simulated_data)
    
    return simulated_df

# Function to process questions with OpenAI's GPT (using system, user, assistant roles)
def chatbot_with_csv(csv_file, question):
    try:
        # Read the uploaded CSV
        csv_data = pd.read_csv(csv_file.name)

        # Define the system message to guide the assistant
        system_message = "You are a helpful assistant that answers questions about network traffic and DDoS attacks. You can also simulate DDoS attacks based on traffic data."

        # Check for specific keywords in the question
        if "simulate" in question.lower():
            # Generate simulated attack data and return as a table
            simulation_result = simulate_attack_data()
            return simulation_result

        elif "predict" in question.lower():
            # Use the models to predict the attack type for the first data point
            result = preprocess_and_predict(csv_file)
            return result.head(1)  # Show prediction for the first data point

        else:
            # Using OpenAI's GPT to answer the question based on the CSV content
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Given the following data:\n{csv_data.head()}\n\nQuestion: {question}"}
            ]

            # Using OpenAI API to get a response
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use the appropriate OpenAI engine
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )

            # Ensure we get the assistant's message correctly
            return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Error processing the file or question: {str(e)}"

# Define the Gradio interface function
def classify_and_simulate(file, question):
    # Make sure we always return three outputs (prediction, simulation, and chatbot response)
    if file is not None:
        if question.strip() == "":
            # Predict from the uploaded file
            result = preprocess_and_predict(file)
            return result, pd.DataFrame(), ""  # Returning empty dataframe for simulation and empty response for the chatbot
        else:
            # Process the question with GPT
            response = chatbot_with_csv(file, question)
            # If it's a simulation, return the simulation data
            if "simulate" in question.lower():
                simulation_result = simulate_attack_data()
                return pd.DataFrame(), simulation_result, response  # Empty prediction and simulation tables
            return pd.DataFrame(), pd.DataFrame(), response  # Empty prediction and simulation tables
    return pd.DataFrame(), pd.DataFrame(), "Please upload a file and ask a question."  # Default output

# Define the Gradio interface
inputs = [
    gr.File(label="Upload a CSV file", elem_id="csv-upload"),  # Custom ID for easier styling
    gr.Textbox(label="Ask a question about the CSV data or DDoS attacks", placeholder="Ask something like 'What is the attack type?' or 'Simulate an attack'", lines=2, elem_id="question-input")  # Multiline question input
]

outputs = [
    gr.Dataframe(headers=["Data Point", "Prediction"], type="pandas", elem_id="prediction-table"),  # Prediction output as a dataframe
    gr.Dataframe(headers=["Attack Type", "Packet Count", "Duration (s)", "Impact Level"], type="pandas", elem_id="simulation-table"),  # Simulate output as a dataframe
    gr.Textbox(label="Chatbot Response", placeholder="Assistant's response will appear here...", elem_id="chat-response", lines=4)  # Chatbot response, multiline
]

# Create the Gradio interface with custom CSS for layout
css = """
    .gradio-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    #question-input {
        width: 100%;
    }

    #chat-response {
        width: 100%;
        margin-top: 15px;
    }
"""

# Create the Gradio interface with nice styling and more interactive design
interface = gr.Interface(
    fn=classify_and_simulate, 
    inputs=inputs, 
    outputs=outputs, 
    live=True,
    title="Intrusion Detection System (IDS) using Machine Learning and Q&A",  # Updated title
    description="Upload a CSV file to classify traffic as 'BENIGN' or 'DDoS' using your trained model. You can also ask questions about the traffic data or simulate DDoS attack traffic. The chatbot will answer based on your question and the uploaded data.",
    theme="default",  # Default theme (no extra theme applied)
    css=css,  # Apply custom CSS
)

# Launch the interface
interface.launch()
