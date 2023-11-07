from uuid import uuid4
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from streamlit_chat import message
import seaborn as sns
from spelltest.entities.managers import MessageType


def render_simulation_results(data):
    df = pd.DataFrame(data["simulations"])
    mean_accuracy = data["aggregated_metrics"]["accuracy"]
    std_accuracy = data["aggregated_metrics"]["accuracy_deviation"]
    # Calculate quantile values

    gauge_col1, gauge_col2 = st.columns(2)

    gauge_col1.metric(label="Mean Accuracy", value=mean_accuracy)
    gauge_col2.metric(label="Std. Dev. Accuracy", value=std_accuracy)

    # Display the gauges
    gauge_col3, gauge_col4, gauge_col5, gauge_col6 = st.columns(4)
    gauge_col3.metric(label="99.9 Quantile",
                      value=data["aggregated_metrics"]["top_999_percentile_accuracy"])
    gauge_col4.metric(label="99 Quantile",
                      value=data["aggregated_metrics"]["top_99_percentile_accuracy"])
    gauge_col5.metric(label="95 Quantile",
                      value=data["aggregated_metrics"]["top_95_percentile_accuracy"])
    gauge_col6.metric(label="50 Quantile",
                      value=data["aggregated_metrics"]["top_50_percentile_accuracy"])

    accuracy_values = [
        eval["accuracy"] for simulation in data["simulations"] for eval in simulation["evaluations"]
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
                if data["chat_mode"]:
                    for i in row["message_storage"]["chat_history"][1:]:
                        message(i["text"], is_user=i["author"] == MessageType.USER.name, key=str(uuid4())[:7])
                else:
                    message(row["message_storage"]["prompt"]["text"], is_user=True, key=str(uuid4())[:7])
                    message(row["message_storage"]["completion"]["text"], is_user=False, key=str(uuid4())[:7])
            with tab2:
                for metric in row["evaluations"]:
                    st.subheader(f"Rationale {metric['metric']['name']}")
                    st.write(metric["rationale"])