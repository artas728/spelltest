import os
from dataclasses import dataclass, field
import urllib.parse
import requests as requests

IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")

@dataclass
class Metric:
    accuracy: float
    accuracy_deviation: float
    top_999_percentile_accuracy: float
    top_99_percentile_accuracy: float
    top_95_percentile_accuracy: float
    top_50_percentile_accuracy: float
    # metric_definition_id: int = None
    rationale: str = None

@dataclass
class MetricDefinition:
    name: str
    definition: str
    db_id: int = field(default=None,
                       init=False)  # Using field to make it clear that db_id shouldn't be passed during initialization

    def __post_init__(self):
        if not IGNORE_DATA_COLLECTING:
            self._synchronise_with_db()

    def _synchronise_with_db(self):
        headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {SPELLFORGE_API_KEY}"}
        encoded_name = urllib.parse.quote(self.name)
        # Get MetricDefinition by name and related user_persona_id
        response = requests.get(f'{SPELLFORGE_HOST}api/metric-definitions/?name={encoded_name}', headers=headers)

        if response.status_code != 200:
            # handle error
            print(f"Error: {response.status_code}")
            return

        data = response.json()

        # If MetricDefinition with the given name and related user_persona doesn't exist, create it.
        if not data:
            create_data = {
                "name": self.name,
                "definition": self.definition,
            }

            create_response = requests.post(f'{SPELLFORGE_HOST}api/metric-definitions/',
                                            json=create_data, headers=headers)

            if create_response.status_code != 201:
                print(f"Error creating MetricDefinition: {create_response.status_code}")
                return

            self.db_id = create_response.json()["id"]  # Set the db_id here after creating
            print("MetricDefinition created successfully!")
            return

        metric_definition = data[0]
        self.db_id = metric_definition["id"]  # Set the db_id here after fetching

        # Check if there are any changes
        changes = {}
        fields_to_check = ["name", "definition"]
        for field in fields_to_check:
            if getattr(self, field) != metric_definition[field]:
                changes[field] = getattr(self, field)

        # If there are changes, update the MetricDefinition
        if changes:
            response = requests.patch(f'{SPELLFORGE_HOST}api/metric-definitions/{self.db_id}/',
                                      json=changes, headers=headers)

            if response.status_code != 200:
                # handle error
                print(f"Error updating: {response.status_code}")
                return

            print("MetricDefinition updated successfully!")
