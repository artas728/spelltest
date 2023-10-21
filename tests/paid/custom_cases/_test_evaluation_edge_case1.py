import os
import pytest
from rich.console import Console
from spelltest.ai_managers.evaluation_manager import EvaluationManager
from spelltest.ai_managers.raw_completion_manager import SyntheticUserCompletionManager
from spelltest.ai_managers.tracing.cost_calculation_tracing import CostCalculationManager
from spelltest.ai_managers.tracing.promtelligence_tracing import PromptelligenceClient
from spelltest.entities.managers import Message, MessageType
from spelltest.entities.metric import MetricDefinition
from spelltest.entities.synthetic_user import SyntheticUserParams, SyntheticUser

case_data = {
    "project_name": "default",
    "aggregated_metrics": {
        "accuracy": 0.8420000000000001,
        "accuracy_deviation": 0.03862926063046172,
        "top_999_percentile_accuracy": 0.869944,
        "top_99_percentile_accuracy": 0.86944,
        "top_95_percentile_accuracy": 0.8672,
        "top_50_percentile_accuracy": 0.8420000000000001,
        "rationale": None
    },
    "simulations": [
        {
            "prompt_version_id": None,
            "app_user_persona_id": None,
            "run_ids": [
                "cfe6ca1f-6d5e-43ce-a21d-906d6e82a1ad",
                "d03cd4f0-9a2c-4232-8d56-46296b0369b6"
            ],
            "length_complexity": 0.0,
            "chat_id": None,
            "evaluations": [
                {
                    "metric": {
                        "name": "tpas",
                        "definition": "TPAS - The Travel Plan Accuracy Score. This metric measures the accuracy of the generated response by evaluating the inclusion of the expected output, well-scheduled travel plan with proper time schedule and nothing else. The TPAS is a numerical value between 0 and 100, with 100 representing a perfect match to the expected output and 0 indicating non-accurate result.",
                        "db_id": None
                    },
                    "accuracy": 0.87,
                    "accuracy_deviation": 0.02449489742783178,
                    "rationale": "Evaluation:\n\n1. Independent Analysis:\nThe AI-generated response acknowledges the user's request for help in planning their first Saturday exploring Seattle. It also expresses enthusiasm in assisting the user and provides an opportunity for the user to specify their preferences and interests. The text is polite and welcoming, creating a positive user experience. \n\n2. Contextual Understanding:\nThe content of the response aligns with the user's expectations of receiving a well-planned, detailed, and comprehensive schedule for their day in Seattle. It acknowledges that planning can be challenging for a busy nomad and offers to tailor the schedule to meet the user's requirements. The response demonstrates an understanding of the user's needs and suggests a personalized approach. \n\n3. Critical Discrepancies:\nThere are no critical discrepancies in the AI-generated response. The response adequately addresses the user's request and sets the stage for creating a well-planned travel schedule. \n\n4. Self-reflective Critique:\nDuring the evaluation process, I did not encounter any significant blind spots or barriers in analyzing the AI-generated response. The response was coherent and aligned with the user's expectations. However, it is important to note that my assessment is based on the available information, and if there were further interactions, the analysis could vary. \n\n5. Constructive Summary:\nThe AI-generated response successfully addresses the user's request for help in planning their first Saturday exploring Seattle. It demonstrates an understanding of the user's needs and offers a personalized approach to create a comprehensive schedule. The response is polite, enthusiastic, and invites the user to provide their preferences. Overall, the response satisfies the user's expectations and aligns well with the provided accuracy definition. \n\nPotential areas for improvement in future AI-generated responses could include providing more specific suggestions or options based on the user's interests and preferences, as well as offering additional information about popular attractions or activities in Seattle that the user may not be aware of. This would enhance the user experience and provide a more detailed and comprehensive travel plan."
                }
            ],
            "message_storage": {
                "prompt": {
                    "author": "USER",
                    "text": "{\n  \"USER_INPUT\": \"I am a busy nomad who just moved to Seattle and wants help planning my first Saturday exploring the city.\"\n}",
                    "run_id": "cfe6ca1f-6d5e-43ce-a21d-906d6e82a1ad"
                },
                "completion": {
                    "author": "ASSISTANT",
                    "text": "Sure! I'd be happy to help you plan your first Saturday exploring Seattle. Seattle is a vibrant city with lots of things to see and do. Do you have any specific interests or preferences that I should take into consideration while planning your day?",
                    "run_id": "d03cd4f0-9a2c-4232-8d56-46296b0369b6"
                },
                "perfect_completion": {
                    "author": "ASSISTANT",
                    "text": "Sure! I understand that as a busy nomad, planning can be challenging for you. Since you're new to Seattle, I'd be happy to help you create a well-planned and detailed schedule for your first Saturday exploring the city. To personalize your experience, could you please let me know your preferences and interests? This will allow me to tailor the schedule to meet your requirements and ensure a comprehensive day of exploring Seattle.",
                    "run_id": "640f6646-b201-4635-8f69-2f99d77c040d"
                }
            },
            "granular_evaluation": False
        },
        {
            "prompt_version_id": None,
            "app_user_persona_id": None,
            "run_ids": [
                "c50bcedf-b150-4499-a63c-c306465f8c6e",
                "027eeef7-128e-4083-b8ce-9770503f5e17"
            ],
            "length_complexity": 0.0,
            "chat_id": None,
            "evaluations": [
                {
                    "metric": {
                        "name": "tpas",
                        "definition": "TPAS - The Travel Plan Accuracy Score. This metric measures the accuracy of the generated response by evaluating the inclusion of the expected output, well-scheduled travel plan with proper time schedule and nothing else. The TPAS is a numerical value between 0 and 100, with 100 representing a perfect match to the expected output and 0 indicating non-accurate result.",
                        "db_id": None
                    },
                    "accuracy": 0.8140000000000001,
                    "accuracy_deviation": 0.052763623833091675,
                    "rationale": "Evaluation:\n\n1. Independent Analysis:\n   - The AI-generated text provides a suggested itinerary for exploring Seattle on a Saturday, starting at 9:00 AM and ending with dinner at 7:00 PM. It includes specific time slots for each activity, such as visiting Pike Place Market, walking along the waterfront, exploring the Seattle Center, visiting the Museum of Pop Culture, strolling through Fremont, and having dinner in Capitol Hill.\n   - The text also mentions specific attractions and points of interest, such as the Pike Place Fish Market, the Seattle Great Wheel, the Space Needle, the Chihuly Garden and Glass exhibit, the Museum of Pop Culture (MoPOP), and the Fremont Troll.\n   - The response encourages the user to adjust the itinerary based on their preferences and interests.\n\n2. Contextual Understanding:\n   - The user's anticipated outcome is to receive a well-planned, detailed, and comprehensive schedule that meets their requirements. The AI-generated text successfully provides a suggested itinerary with specific activities and time slots, meeting the user's expectations in terms of detail and comprehensiveness.\n   - The text accurately presents popular attractions in Seattle, such as Pike Place Market, the waterfront, the Seattle Center, and the Museum of Pop Culture. It also suggests iconic experiences like watching the fish throwing at Pike Place Market and visiting the Fremont Troll.\n   - The response acknowledges the user's struggle with planning and aims to provide a helpful and comprehensive schedule, addressing the user's needs and requirements.\n\n3. Critical Discrepancies:\n   - The AI-generated text lacks some specific details that could enhance the user's experience. For example, it does not mention the duration of each activity or provide any transportation recommendations or directions between different locations. This information could be valuable for a busy nomad who may not be familiar with the city.\n   - The text does not explicitly address the user's struggle with planning, as stated in their input. While it provides a well-planned itinerary, it would be helpful for the AI to acknowledge the user's initial challenge and assure them that it will provide a detailed schedule.\n   - The AI-generated text could also benefit from including more personalized recommendations based on the user's preferences or offering alternative activities for different interests. This would further enhance the user's experience and make the itinerary more tailored to their needs.\n\n4. Self-reflective Critique:\n   - One potential blind spot in this evaluation is the assumption that the AI-generated text is solely responsible for planning the user's first Saturday exploration in Seattle. The user may have other sources of information or preferences that they can incorporate into their plans, and the AI-generated text may serve as a valuable starting point rather than a definitive schedule.\n   - Another subjective interpretation is the assessment of the level of detail and comprehensiveness in the AI-generated text. While it does provide specific activities and time slots, it may still lack certain aspects that the user considers important. Therefore, the evaluation should take into account the user's subjective expectations to avoid overlooking potential areas for improvement.\n\n5. Constructive Summary:\n   - The AI-generated text successfully provides a well-planned itinerary for exploring Seattle on a Saturday, meeting the user's expectations for a detailed and comprehensive schedule.\n   - However, there are opportunities for improvement. The text could include specific durations for each activity, transportation recommendations or directions, and acknowledge the user's initial struggle with planning. It could also offer more personalized recommendations or alternative activities based on the user's preferences to enhance their experience.\n   - To enhance future AI-generated responses, it would be beneficial to incorporate more contextual understanding of the user's specific challenges and provide a more personalized and tailored approach to planning. Additionally, including more specific details and guidance in the itinerary would further improve the user's experience and satisfaction."
                }
            ],
            "message_storage": {
                "prompt": {
                    "author": "USER",
                    "text": "{\n  \"USER_INPUT\": \"You're a very busy nomad who struggles with planning. You're moved to Seattle and looking at how to spend your first Saturday exploring the city\"\n}",
                    "run_id": "c50bcedf-b150-4499-a63c-c306465f8c6e"
                },
                "completion": {
                    "author": "ASSISTANT",
                    "text": "Sure! I can help you with that. Seattle has a lot to offer, so let's plan your first Saturday exploration. Here's a suggested itinerary for you:\n\n9:00 AM - Start your day with a visit to the Pike Place Market. This iconic Seattle attraction is bustling with local vendors, fresh produce, and unique crafts. Don't forget to watch the famous fish throwing at the Pike Place Fish Market.\n\n10:30 AM - Take a walk along the waterfront. Enjoy the beautiful views of Elliott Bay and the Olympic Mountains. You can also stop by the Seattle Great Wheel for a fun ride and panoramic views of the city.\n\n12:00 PM - Treat yourself to a delicious seafood lunch at one of the waterfront restaurants. I highly recommend trying out Ivar's Acres of Clams or Anthony's Pier 66 for a quintessential Seattle seafood experience.\n\n1:30 PM - Explore the Seattle Center. Visit the iconic Space Needle and take the elevator to the observation deck for a stunning 360-degree view of the city. You can also check out the Chihuly Garden and Glass exhibit nearby for some incredible glass art.\n\n3:00 PM - Head to the Museum of Pop Culture (MoPOP) located within the Seattle Center. This unique museum showcases various aspects of popular culture, including music, film, and science fiction. It's a great place to immerse yourself in creativity and entertainment.\n\n5:00 PM - Take a stroll through the quirky neighborhood of Fremont. Check out the famous Fremont Troll, a large sculpture under the bridge, and explore the eclectic shops and cafes in the area. Don't miss the Fremont Sunday Market if you're interested in vintage treasures and local vendors.\n\n7:00 PM - End your day with a memorable dinner at one of the trendy restaurants in Capitol Hill. This vibrant neighborhood offers a wide range of dining options, from casual eateries to upscale establishments. Some popular choices include Spinasse, Ba Bar, and Stateside.\n\nThat's just a sample itinerary to help you get started. Feel free to adjust it based on your preferences and interests. Enjoy exploring Seattle!",
                    "run_id": "027eeef7-128e-4083-b8ce-9770503f5e17"
                },
                "perfect_completion": {
                    "author": "ASSISTANT",
                    "text": "Sure! I can understand how challenging it can be to plan your exploration in a new city. Seattle has a lot to offer and I'm here to help you come up with a well-planned schedule for your first Saturday. \n\nBased on your nomadic lifestyle and busy schedule, here's a comprehensive itinerary for you:\n\n9:00 AM - Start your day with a visit to the Pike Place Market. This bustling market offers a wide variety of local vendors, fresh produce, and unique crafts. Don't miss the famous fish throwing at the Pike Place Fish Market.\n\n10:30 AM - Enjoy a leisurely walk along the waterfront. Take in the beautiful views of Elliott Bay and the Olympic Mountains. You may also want to include a fun ride on the Seattle Great Wheel for panoramic views of the city.\n\n12:00 PM - Treat yourself to a delicious seafood lunch at one of the waterfront restaurants. I recommend trying out Ivar's Acres of Clams or Anthony's Pier 66 for an authentic Seattle seafood experience.\n\n1:30 PM - Explore the Seattle Center, where you'll find attractions like the iconic Space Needle. Take the elevator to the observation deck for stunning 360-degree views of the city. Nearby, the Chihuly Garden and Glass exhibit showcases incredible glass art.\n\n3:00 PM - Head to the Museum of Pop Culture (MoPOP) located within the Seattle Center. This unique museum celebrates various aspects of popular culture, including music, film, and science fiction. It's a great place to immerse yourself in creativity and entertainment.\n\n5:00 PM - Take a leisurely stroll through the quirky neighborhood of Fremont. Don't miss the famous Fremont Troll sculpture under the bridge, and explore the eclectic shops and cafes in the area. If you're interested, the Fremont Sunday Market offers vintage treasures and local vendors.\n\n7:00 PM - End your day with a memorable dinner at one of the trendy restaurants in Capitol Hill. This vibrant neighborhood offers a wide range of dining options. Some popular choices include Spinasse, Ba Bar, and Stateside.\n\nI've made sure to consider your busy lifestyle and desire for a detailed schedule. Feel free to adjust this itinerary based on your preferences and interests. Let me know if you need any further assistance. Enjoy exploring Seattle!",
                    "run_id": "c7baa2bc-96d7-48b7-8d5d-eaeac3a083a5"
                }
            },
            "granular_evaluation": False
        }
    ],
    "llm_name": "gpt-3.5-turbo",
    "size": 2,
    "chat_mode": False,
    "temperature": 0.7,
    "reason": "MANUAL",
    "reason_value": "5e3c8599-7938-4a25-a52a-08035a3989c1",
    "status": "SUCCESS"
}

IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
console = Console()
cost_calculation_manager = CostCalculationManager(console=console)
tracing = PromptelligenceClient(ignore=IGNORE_DATA_COLLECTING)

@pytest.fixture
def setup_completion():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm_name = "gpt-3.5-turbo"
    TARGET_PROMPT = """
    As a travel planner, your goal is to create a detailed travel plan for a traveler based on their requirements and preferences. You should carefully review the traveler's description and travel requirements and create a comprehensive itinerary that includes a detailed plan for each hour of their trip, taking into account transportation times and potential risks.
Your travel plan should consider the traveler's desired activities, attractions, and destinations, and outline the most efficient and convenient routes and modes of transportation to maximize their time and minimize travel disruptions.
In addition, you should prioritize the safety and security of the traveler by considering potential risks and recommending precautions. This may include suggesting alternative routes or modes of transportation to avoid high-risk areas, providing safety tips, or recommending travel insurance if necessary.
Please provide a well-structured and organized travel plan that includes specific details for each hour of the trip, such as the activities or attractions to visit, the transportation modes and routes, estimated travel times, and any potential risks or safety considerations.
Your response should be creative, flexible, and tailored to the traveler's preferences and requirements, while also ensuring accuracy, feasibility, and a focus on potential risks and safety.
    """
    user_params = SyntheticUserParams(
        temperature=0.7,
        llm_name="gpt-3.5-turbo",
        description="You're a very busy nomad who struggles with planning. You're moved to Seattle and looking at how to spend your first Saturday exploring the city",
        expectation="Well-planned objective, detailed, and comprehensive schedule that meets user's requirements",
        user_knowledge_about_app="The app receives text input about travel requirements (i.e., place, preferences, short description of the family and their interests) and returns a travel schedule that accommodates all family members’ needs and interests."
    )
    metric_definition = MetricDefinition(
                name="tpas",
                definition="TPAS - The Travel Plan Accuracy Score. This metric measures the accuracy of the generated response by evaluating the inclusion of the expected output, well-scheduled travel plan with proper time schedule and nothing else. The TPAS is a numerical value between 0 and 100, with 100 representing a perfect match to the expected output and 0 indicating non-accurate result."
            )
    user = SyntheticUser(
        name="Busy Nomad in Seattle",
        params=user_params,
        metrics=[
            metric_definition
        ]
    )
    user_completion_manager = SyntheticUserCompletionManager(user, TARGET_PROMPT, openai_api_key)
    manager = EvaluationManager(
        metric_definitions=[metric_definition],
        openai_api_key=openai_api_key,
        synthetic_user_persona_manager=user_completion_manager,
        llm_name_default=llm_name,
    )
    manager.initialize_evaluation()
    return manager


@pytest.mark.asyncio
async def test_unfinished_result_evaluation_with_failed_perfect_completion(setup_completion):
    """ make sure evaluation rationale represed it as failed case"""
    prompt = Message(
        author=MessageType.USER,
        text="{\n  \"USER_INPUT\": \"I am a busy nomad who just moved to Seattle and wants help planning my first Saturday exploring the city.\"\n}",
    )
    completion = Message(   # failed completion
        author=MessageType.ASSISTANT,
        text="Sure! I'd be happy to help you plan your first Saturday exploring Seattle. Seattle is a vibrant city with lots of things to see and do. Do you have any specific interests or preferences that I should take into consideration while planning your day?"
    )
    # perfect_completion = Message(   # failed perfect completion
    #     author=MessageType.ASSISTANT,
    #     text="Sure! I understand that as a busy nomad, planning can be challenging for you. Since you're new to Seattle, I'd be happy to help you create a well-planned and detailed schedule for your first Saturday exploring the city. To personalize your experience, could you please let me know your preferences and interests? This will allow me to tailor the schedule to meet your requirements and ensure a comprehensive day of exploring Seattle."
    # )
    result = await setup_completion._evaluate("USER:\n"+prompt.text+completion.text)
    for evaluation in result:
        assert type(evaluation.rationale) is str
        assert type(evaluation.accuracy) is float
        assert 0 <= evaluation.accuracy <= 0.5
        assert type(evaluation.accuracy_deviation) is float
        assert 0 <= evaluation.accuracy_deviation <= 0.2



@pytest.mark.asyncio
async def test_unfinished_result_evaluation_with_good_perfect_completion(setup_completion):
    """ make sure evaluation rationale represed it as failed case"""
    prompt = Message(
        author=MessageType.USER,
        text="{\n  \"USER_INPUT\": \"I am a busy nomad who just moved to Seattle and wants help planning my first Saturday exploring the city.\"\n}",
    )
    completion = Message(   # failed completion
        author=MessageType.ASSISTANT,
        text="Sure! I'd be happy to help you plan your first Saturday exploring Seattle. Seattle is a vibrant city with lots of things to see and do. Do you have any specific interests or preferences that I should take into consideration while planning your day?"
    )
    # perfect_completion = Message(   # good perfect completion
    #     author=MessageType.ASSISTANT,
    #     text="Sure! I can help you with that. Seattle has a lot to offer, so let's plan your first Saturday exploration. Here's a suggested itinerary for you:\n\n9:00 AM - Start your day with a visit to the Pike Place Market. This iconic Seattle attraction is bustling with local vendors, fresh produce, and unique crafts. Don't forget to watch the famous fish throwing at the Pike Place Fish Market.\n\n10:30 AM - Take a walk along the waterfront. Enjoy the beautiful views of Elliott Bay and the Olympic Mountains. You can also stop by the Seattle Great Wheel for a fun ride and panoramic views of the city.\n\n12:00 PM - Treat yourself to a delicious seafood lunch at one of the waterfront restaurants. I highly recommend trying out Ivar's Acres of Clams or Anthony's Pier 66 for a quintessential Seattle seafood experience.\n\n1:30 PM - Explore the Seattle Center. Visit the iconic Space Needle and take the elevator to the observation deck for a stunning 360-degree view of the city. You can also check out the Chihuly Garden and Glass exhibit nearby for some incredible glass art.\n\n3:00 PM - Head to the Museum of Pop Culture (MoPOP) located within the Seattle Center. This unique museum showcases various aspects of popular culture, including music, film, and science fiction. It's a great place to immerse yourself in creativity and entertainment.\n\n5:00 PM - Take a stroll through the quirky neighborhood of Fremont. Check out the famous Fremont Troll, a large sculpture under the bridge, and explore the eclectic shops and cafes in the area. Don't miss the Fremont Sunday Market if you're interested in vintage treasures and local vendors.\n\n7:00 PM - End your day with a memorable dinner at one of the trendy restaurants in Capitol Hill. This vibrant neighborhood offers a wide range of dining options, from casual eateries to upscale establishments. Some popular choices include Spinasse, Ba Bar, and Stateside.\n\nThat's just a sample itinerary to help you get started. Feel free to adjust it based on your preferences and interests. Enjoy exploring Seattle!",
    # )
    result = await setup_completion._evaluate("USER:\n"+prompt.text+completion.text)
    for evaluation in result:
        assert type(evaluation.rationale) is str
        assert type(evaluation.accuracy) is float
        assert 0 <= evaluation.accuracy <= 0.5
        assert type(evaluation.accuracy_deviation) is float
        assert 0 <= evaluation.accuracy_deviation <= 0.2



# @pytest.mark.asyncio
# async def test_accuracy(setup_manager_chat):
#     ...
    # accuracy, deviation = await setup_manager_chat._accuracy('<...all simulation text..>')
    # assert type(accuracy) is float
    # assert 0 < accuracy < 1.0
    # assert type(deviation) is float
    # assert 0 < deviation < 1.0
#
# @pytest.mark.asyncio
# async def test_perfect_completion(setup_manager_completion):
#     prompt = Message(
#         author=MessageType.USER,
#         text="text"
#     )
#     completion = Message(
#         author=MessageType.ASSISTANT,
#         text="text"
#     )
#     perfect_message: Message = await setup_manager_completion._generate_perfect_completion(
#         prompt,
#         completion
#     )
#
#     assert type(perfect_message.text) is str
#     assert perfect_message.author == MessageType.ASSISTANT
#
# @pytest.mark.asyncio
# async def test_perfect_chat_message(setup_manager_chat):
#     chat_history = [
#         Message(
#             author=MessageType.USER,
#             text="text"
#         ),
#         Message(
#             author=MessageType.ASSISTANT,
#             text="text2"
#         )
#     ]
#     perfect_message: Message = await setup_manager_chat._generate_perfect_chat_message(
#         Message(
#             author=MessageType.USER,
#             text="perfect text3"
#         ),
#         chat_history
#     )
#
#     assert type(perfect_message.text) is str
#     assert perfect_message.author == MessageType.ASSISTANT
#
#
# @pytest.mark.asyncio
# async def test_evaluate(setup_manager_chat):
#     chat_history = [
#         Message(
#             author=MessageType.USER,
#             text="text"
#         ),
#         Message(
#             author=MessageType.ASSISTANT,
#             text="text2"
#         )
#     ]
#     perfect_chat_history = [
#         Message(
#             author=MessageType.USER,
#             text="text"
#         ),
#         Message(
#             author=MessageType.ASSISTANT,
#             text="text2"
#         )
#     ]
#
#     result = await setup_manager_chat._evaluate(chat_history, perfect_chat_history)
#     for evaluation in result:
#         assert type(evaluation.rationale) is str
#         assert type(evaluation.accuracy) is float
#         assert 0 <= evaluation.accuracy <= 1.0
#         assert type(evaluation.deviation) is float
#         assert 0 <= evaluation.deviation <= 1.0