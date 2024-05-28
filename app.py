import streamlit as st
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os

st.set_page_config(layout="wide", page_title="LocalLama Benchmark", page_icon="🦙")

cpu, gpu, ollama_models = {}, {}, {}
with open("data/cpu.json", "r") as f:
    cpu = json.load(f)  # read cpu.json into a dictionary
with open("data/gpu.json", "r") as f:
    gpu = json.load(f)  # read gpu.json into a dictionary
with open("data/ollama_models.json", "r") as f:
    ollama_models = json.load(f)  # read ollama_models.json into a dictionary

# Title
title_cols = st.columns([3, 2])
title_cols[0].title("LocalLlama Benchmark")
title_cols[0].markdown(
    """
LocalLlama Benchmark is a benchmarking tool that provides insights into the performance of various machine learning models on different hardware configurations. 

By leveraging LocalLlama, you can compare the performance of models on CPU and GPU, helping you make informed decisions about the optimal hardware configuration for your specific use case.

[WE ARE CURRENTLY COLLECTING DATA. AFTER SUFFICIENT DATA IS COLLECTED, WE WILL PROVIDE THE INSIGHTS]

**Note: all data is open-source and shared with the community.**
"""
)
title_cols[1].image(
    "https://raw.githubusercontent.com/0ssamaak0/LocallamaBenchMark/main/assets/LlBN.png",
    width=350,
)

st.subheader("Machine Specs")
# Operating System
os_options = ["Linux", "MacOS", "Windows", "Windows: WSL"]
os_type = st.selectbox("Operating System", os_options)

# CPU
# Create two columns
col1, col2 = st.columns([5, 6])

# CPU Brand select box in the first column
cpu_brand = col1.selectbox(
    "Processor Brand", list(cpu.keys()), key="cpu_brand", index=2
)

# CPU Model select box in the second column
cpu_models = cpu[cpu_brand]
cpu_model = col2.selectbox("Processor Model", cpu_models, key="cpu_model", index=2)

# GPU
if cpu_brand == "Apple":
    gpu_brand = cpu_brand
    gpu_model = cpu_model
else:
    col1, col2, col3 = st.columns([5, 5, 1])
    gpu_brand = col1.selectbox(
        "GPU Brand", options=list(gpu.keys()) + ["None"], index=2
    )
    if gpu_brand != "None":
        gpu_models = gpu[gpu_brand]
        gpu_models = gpu[gpu_brand]
        gpu_model = col2.selectbox("GPU Model", options=gpu_models)
        gpu_brand_index = list(gpu.keys()).index(gpu_brand)
        gpu_model_index = gpu_models.index(gpu_model)
        num_gpus = col3.number_input(
            "Number of GPUs", value=1, min_value=1, max_value=16
        )
    else:
        gpu_model = "None"
        gpu_model = col2.selectbox("GPU Model", options=["None"], disabled=True)
        num_gpus = col3.number_input("Number of GPUs", value=0, disabled=True)

# RAM
ram = st.number_input("RAM (GB)", min_value=2, max_value=2**40, value=16, step=2)


# Models
st.subheader("Models Inference Speed")
# Initialize the session state
if "stage" not in st.session_state:
    st.session_state.stage = 0

if "models_count" not in st.session_state:
    st.session_state.models_count = 0


def set_stage(stage):
    st.session_state.stage = stage


# Step 1: Form to determine the number of forms
with st.form(key="Number of Models Tested"):
    n_forms = st.number_input("Number of Models Tested", 0)
    submit_button = st.form_submit_button(label="Start", on_click=set_stage, args=(1,))

# Step 2: Generate forms dynamically based on user input
if st.session_state.stage > 0:
    if "models" not in st.session_state:
        st.session_state.models = []

    current_models = []
    for i in range(int(n_forms)):
        with st.container():
            cols = st.columns(5)
            model_name = cols[0].selectbox(
                "Model name",
                list(
                    ollama_models.keys()
                ),  # Assuming ollama_models is defined elsewhere
                key=f"model_name_{i}",
                placeholder="Select your model",
                index=0,
            )
            model_sizes = list(ollama_models[model_name].keys())
            num_params = cols[1].selectbox(
                "Number of Parameters (Billions)",
                options=model_sizes,
                key=f"nparams_{i}",
                index=0,
            )
            quant_methods = list(ollama_models[model_name][num_params])
            quant_method = cols[2].selectbox(
                "Quantization Method",
                options=quant_methods,
                key=f"quantization_method_{i}",
                index=quant_methods.index("Q4_0"),
            )
            input_rate = cols[3].number_input(
                "Input Prompt rate (Tokens/s)",
                key=f"input_token_{i}",
                min_value=0.0,
                step=1.0,
                format="%f",
                value=20.0,
            )
            output_rate = cols[4].number_input(
                "Output Completion rate (Token/s)",
                key=f"output_token_{i}",
                min_value=0.0,
                step=1.0,
                format="%f",
                value=20.0,
            )

            # Collect the current form data
            current_models.append(
                {
                    "model_name": model_name,
                    "num_params": num_params,
                    "quant_method": quant_method,
                    "input_tokens": input_rate,
                    "output_tokens": output_rate,
                }
            )


# Submit button
if st.button("Submit"):
    st.session_state.models = current_models
    set_stage(2)
    # Check if the file exists and read the data
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new data
    data.append(
        {
            "Operating System": os_type,
            "Processor": cpu_model,
            "GPU": gpu_model,
            "Number of GPUs": num_gpus,
            "RAM": f"{ram} GB",
            "Models": st.session_state.models,
        }
    )

    # Write the data back to the file
    with open("results.json", "w") as f:
        json.dump(data, f)

    st.success("Information submitted and written to results.json")


# read data
@st.cache_data
def get_results(path="results.json"):
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


results = get_results()
# Download Raw Data
st.download_button(
    label="Download data as JSON",
    data=Path("results.json").read_text(),
    file_name="LocalLlamaBenchmark.json",
)
