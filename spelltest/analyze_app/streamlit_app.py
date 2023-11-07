import json
import os
import streamlit as st

from spelltest.analyze_app.streamlit_simulations import render_simulation_results


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
                render_simulation_results(simulation_session_data)


if __name__ == "__main__":
    main()
