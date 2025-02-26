import requests
from openai import OpenAI
import pandas as pd
import re
import time
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")
lmunit_api_key = os.environ.get("LMUNIT_API_KEY")
# Set your OpenAI API key
client = OpenAI(api_key=openai_api_key)

# Global unit tests apply to all queries in your evaluation set
global_unit_tests = ["Is the response a complete sentence?"]


def generate_llm_output(query: str):
    """
    Use your LLM to generate a response to the given query. In this example, we use an off-the-shelf LLM.
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": query}],
        temperature=0,
    )

    return response.choices[0].message.content


def lmunit(query: str, response: str, unit_test: str):
    """
    Use LMUnit to evaluate the response against the given unit test.
    """
    params = {"query": query, "response": response, "unit_test": unit_test}
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {lmunit_api_key}",
        "Content-Type": "application/json",
    }
    lmunit_api_url = "https://api.contextual.ai/v1/lmunit"

    unit_test_score = requests.post(lmunit_api_url, json=params, headers=headers)

    try:
        print(unit_test_score.json())
        return unit_test_score.json()["score"]
    except:
        print(unit_test_score.json())


def test_evalset():
    """
    This method retrieves queries, responses, and unit tests from an evalset.csv file. Then, for each query, it runs each unit test.
    Alternatively, you could use the @pytest.mark.parametrize method.
    """

    # Read the CSV file
    df = pd.read_csv("evalset.csv")

    # Read each column individually
    queries = df["query"].tolist()
    responses = df["response"].tolist()
    unit_tests = df["unit_tests"].tolist()

    # run unit tests on each column
    for query, response, unit_test_row_string in zip(queries, responses, unit_tests):
        print(query)
        print(response)
        unit_test_row_list = re.findall(
            r"“([^”]*)”", unit_test_row_string
        )  # there has to be a better way
        unit_test_full_list = global_unit_tests + unit_test_row_list
        for unit_test in unit_test_full_list:
            print(unit_test)
            unit_test_score = lmunit(query, response, unit_test)
            assert unit_test_score >= 1
            time.sleep(1)
        print("-----")


if __name__ == "__main__":  # TODO: Remove all of this

    query = "What is the capital of France?"
    response = "Capital of France is Paris."
    unit_test = "Does the response correctly identify the capital of France?"

    # unit_test_score = lmunit(query, response, unit_test)
    # print(unit_test_score)

    test_evalset()
