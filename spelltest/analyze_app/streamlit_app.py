import json
import os
from uuid import uuid4
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from streamlit_chat import message
import seaborn as sns
from spelltest.entities.managers import MessageType


def discover_projects(base_dir):
    projects = []
    for root, dirs, files in os.walk(base_dir):
        if 'spelltest_result' in dirs:
            projects.append(os.path.join(root, 'spelltest_result'))
    return projects

def main():
    st.sidebar.title("Spelltest explorer")

    base_dir = '.'  # Start searching from the current directory
    project_paths = discover_projects(base_dir)

    # Combine all projects into a single list of choices
    all_project_choices = []
    for project_path in project_paths:
        project_choices = [os.path.join(project_path, d) for d in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, d))]
        all_project_choices.extend(project_choices)

    project = st.sidebar.selectbox("Select project:", sorted(all_project_choices))

    # If a project is selected, retrieve the list of simulation sessions (JSON files)
    if project:
        simulation_sessions_path = project  # Directly using 'project' since it's a full path
        simulation_session_choices = sorted([f for f in os.listdir(simulation_sessions_path) if f.endswith('.json')], reverse=True)
    else:
        simulation_session_choices = []

    simulation_session = st.sidebar.selectbox("Select simulation session:", simulation_session_choices)

    # If a simulation session is selected, load and display the data
    if simulation_session:
        show_simulations_button = st.sidebar.button("Show Simulations")
        if show_simulations_button:
            with st.spinner("Uploading simulations..."):
                st.text(f"file: {project}/{simulation_session}")
                # Build the path to the chosen JSON file
                json_path = os.path.join(project, simulation_session)  # Directly using 'project' as base path

                # Load the JSON data
                with open(json_path, 'r') as f:
                    simulation_session_data = json.load(f)
                df = pd.DataFrame(simulation_session_data["simulations"])
                mean_accuracy = simulation_session_data["aggregated_metrics"]["accuracy"]
                std_accuracy = simulation_session_data["aggregated_metrics"]["accuracy_deviation"]
                # Calculate quantile values

                gauge_col1, gauge_col2 = st.columns(2)

                gauge_col1.metric(label="Mean Accuracy", value=mean_accuracy)
                gauge_col2.metric(label="Std. Dev. Accuracy", value=std_accuracy)

                # Display the gauges
                gauge_col3, gauge_col4, gauge_col5, gauge_col6 = st.columns(4)
                gauge_col3.metric(label="99.9 Quantile", value=simulation_session_data["aggregated_metrics"]["top_999_percentile_accuracy"])
                gauge_col4.metric(label="99 Quantile", value=simulation_session_data["aggregated_metrics"]["top_99_percentile_accuracy"])
                gauge_col5.metric(label="95 Quantile", value=simulation_session_data["aggregated_metrics"]["top_95_percentile_accuracy"])
                gauge_col6.metric(label="50 Quantile", value=simulation_session_data["aggregated_metrics"]["top_50_percentile_accuracy"])

                accuracy_values = [
                    eval["accuracy"] for simulation in simulation_session_data["simulations"] for eval in simulation["evaluations"]
                ]

                # Visualizing the Gaussian distribution
                plt.figure(figsize=(10, 6))
                plt.style.use('dark_background')  # this will automatically set the background of the plot to black
                sns.histplot(accuracy_values, kde=True, stat="density", bins=10, linewidth=0)
                plt.title('Gaussian Distribution of Accuracy')
                plt.xlabel('Accuracy')
                plt.ylabel('Density')
                plt.grid(axis='y')

                # Display the plot in Streamlit
                st.pyplot(plt)

                # Create expandable row UI
                for i, row in df.iterrows():
                    expand_label = f"Simulation {i} (Accuracy {', '.join([str(metric['accuracy']) for metric in row['evaluations']])})"
                    with st.expander(expand_label, expanded=False):
                        for metric in row["evaluations"]:
                            st.subheader(f"Metric '{metric['metric']['name']}'")
                        gauge_col1, gauge_col2 = st.columns(2)

                        gauge_col1.metric(label="Mean Accuracy", value=metric["accuracy"])
                        gauge_col2.metric(label="Std. Dev. Accuracy", value=metric["accuracy_deviation"])
                        tab1, tab2 = st.tabs(["Chat", "Rationale"])
                        with tab1:
                            if simulation_session_data["chat_mode"]:
                                for i in row["message_storage"]["chat_history"][1:]:
                                    message(i["text"], is_user=i["author"] == MessageType.USER.name, key=str(uuid4())[:7])
                            else:
                                message(row["message_storage"]["prompt"]["text"], is_user=True, key=str(uuid4())[:7])
                                message(row["message_storage"]["completion"]["text"], is_user=False, key=str(uuid4())[:7])
                        with tab2:
                            for metric in row["evaluations"]:
                                st.subheader(f"Rationale {metric['metric']['name']}")
                                st.write(metric["rationale"])


if __name__ == "__main__":
    main()
