import os
import json
import pandas as pd
import openai
from prompts.pygod_ms_prompt import generate_model_selection_prompt as pygod_generate_model_selection_prompt
from prompts.pyod_ms_prompt import generate_model_selection_prompt as pyod_generate_model_selection_prompt
from prompts.orion_ms_prompt import generate_model_selection_prompt as orion_generate_model_selection_prompt
from config import PrivacyConfig

privacy_config = PrivacyConfig()


def init_gpt():
    client = openai.OpenAI(
        organization=privacy_config.organization,
        project=privacy_config.project,
        api_key=privacy_config.gpt_api_key
    )
    return client


def run_gpt(gpt_client, prompt):
    try:
        response = gpt_client.chat.completions.create(
            model="o4-mini",
            messages=prompt
        )
    except Exception as e:
        print(f"Invalid request: {e}")
        return None
    
    return response.choices[0].message.content


def test_pyod_model_selection(pyod_df, pyod_output_path, gpt_client):
    with open(pyod_output_path, 'w') as f:
        # iterate over the rows of the dataframe
        for _, row in pyod_df.iterrows():
            # get the values of the row
            name = row['name']
            size = row['size']
            dim = row['dim']
            # generate the model selection prompt
            prompt = pyod_generate_model_selection_prompt(name, size, dim)
            # run the gpt model
            print(f"Running model selection for {name}...")
            response = run_gpt(gpt_client, prompt)
            # extract "choice" from the response
            try:
                response_json = json.loads(response)
                choice = response_json.get("choice", "No choice found")
            except json.JSONDecodeError:
                choice = "Invalid JSON response"
            # write response and choice to file
            f.write(f"Response for {name}:\n{response}\n{choice}\n---------------------\n")
            

def test_pygod_model_selection(pygod_df, pygod_output_path, gpt_client):
    with open(pygod_output_path, 'w') as f:
        # iterate over the rows of the dataframe
        for _, row in pygod_df.iterrows():
            # get the values of the row
            name = row['name']
            num_node = row['num_node']
            num_edge = row['num_edge']
            num_feature = row['num_feature']
            avg_degree = row['avg_degree']
            # generate the model selection prompt
            prompt = pygod_generate_model_selection_prompt(name, num_node, num_edge, num_feature, avg_degree)
            # run the gpt model
            print(f"Running model selection for {name}...")
            response = run_gpt(gpt_client, prompt)
            # extract "choice" from the response
            try:
                response_json = json.loads(response)
                choice = response_json.get("choice", "No choice found")
            except json.JSONDecodeError:
                choice = "Invalid JSON response"
            # write response and choice to file
            f.write(f"Response for {name}:\n{response}\n{choice}\n---------------------\n")

            
def test_orion_model_selection(orion_df, orion_output_path, gpt_client):
    with open(orion_output_path, 'w') as f:
        # iterate over the rows of the dataframe
        for _, row in orion_df.iterrows():
            # get the values of the row
            name = row['name']
            num_signals = row['num_signals']
            # generate the model selection prompt
            prompt = orion_generate_model_selection_prompt(name, num_signals)
            # run the gpt model
            print(f"Running model selection for {name}...")
            response = run_gpt(gpt_client, prompt)
            # extract "choice" from the response
            try:
                response_json = json.loads(response)
                choice = response_json.get("choice", "No choice found")
            except json.JSONDecodeError:
                choice = "Invalid JSON response"
            # write response and choice to file
            f.write(f"Response for {name}:\n{response}\n{choice}\n---------------------\n")


def main():
    index = 1
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, 'ms_exp_data')
    ms_results_dir = os.path.join(cur_dir, 'ms_results')
    # pyod_df = pd.read_csv(os.path.join(data_dir, 'pyod.csv'))
    # pyod_output_path = os.path.join(ms_results_dir, f'pyod_results_o4mini_{index}.txt')
    # pygod_df = pd.read_csv(os.path.join(data_dir, 'pygod.csv'))
    # pygod_output_path = os.path.join(ms_results_dir, f'pygod_results_o4mini_{index}.txt')
    orion_df = pd.read_csv(os.path.join(data_dir, 'orion.csv'))
    orion_output_path = os.path.join(ms_results_dir, f'orion_results_o4mini_{index}.txt')

    gpt_client = init_gpt()
    # print("Testing PyOD model selection...")
    # test_pyod_model_selection(pyod_df, pyod_output_path, gpt_client)

    # print("======================")

    # print("Testing PyGOD model selection...")
    # test_pygod_model_selection(pygod_df, pygod_output_path, gpt_client)

    print("======================")
    print("Testing Orion model selection...")
    test_orion_model_selection(orion_df, orion_output_path, gpt_client)

if __name__ == "__main__":
    main()