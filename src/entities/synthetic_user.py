import os
import urllib

import requests as requests
from dataclasses import dataclass, field
from .metric import MetricDefinition
from typing import List

IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")

@dataclass
class SyntheticUserParams:
    temperature: float
    llm_name: str
    description: str
    expectation: str
    user_knowledge_about_app: str

@dataclass
class SyntheticUser:
    name: str
    params: SyntheticUserParams
    metrics: List[MetricDefinition]
    db_id: int = field(default=None,
                       init=False)  # Using field to make it clear that db_id shouldn't be passed during initialization

    def __post_init__(self):
        if not IGNORE_DATA_COLLECTING:
            self._synchronise_with_db()

    def _synchronise_with_db(self):
        headers = {"Authorization": f"Api-Key {SPELLFORGE_API_KEY}"}
        encoded_name = urllib.parse.quote(self.name)
        response = requests.get(f'{SPELLFORGE_HOST}api/app-user-personas/?name={encoded_name}', headers=headers)

        if response.status_code != 200:
            # handle error
            print(f"Error: {response.status_code}")
            return

        data = response.json()

        # If AppUserPersona with the given name doesn't exist, create it.
        if not data:
            create_data = {
                "name": self.name,
                "description": self.params.description,
                "expectation": self.params.expectation,
                "user_knowledge_about_app": self.params.user_knowledge_about_app
            }

            create_response = requests.post(f'{SPELLFORGE_HOST}api/app-user-personas/',
                                            json=create_data, headers=headers)

            if create_response.status_code != 201:
                print(f"Error creating AppUserPersona: {create_response.status_code}")
                return

            print("AppUserPersona created successfully!")
            self.db_id = create_response.json()["id"]  # Set the db_id here after creating
            return

        app_user_persona = data[0]
        self.db_id = app_user_persona["id"]  # Set the db_id here after fetching

        # Check if there are any changes
        changes = {}
        fields_to_check = ["description", "expectation", "user_knowledge_about_app"]
        for field in fields_to_check:
            if getattr(self.params, field) != app_user_persona[field]:
                changes[field] = getattr(self.params, field)

        # If there are changes, update the AppUserPersona
        if changes:
            response = requests.patch(f'{SPELLFORGE_HOST}api/app-user-personas/{app_user_persona["id"]}/',
                                      json=changes, headers=headers)

            if response.status_code != 200:
                # handle error
                print(f"Error updating: {response.status_code}")
                return

            print("AppUserPersona updated successfully!")