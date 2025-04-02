import requests
import pandas as pd
import time
import os

lmunit_api_key = os.environ.get("LMUNIT_API_KEY")

# Global unit tests apply to all queries in your evaluation set
global_unit_tests = ["Is the response a complete sentence?"]


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

    return unit_test_score.json()["score"]


def test_evalset():
    """
    This method retrieves queries, responses, and unit tests from an evalset.jsonl file. Then, for each query, it runs each unit test.
    Alternatively, you could use the @pytest.mark.parametrize method.
    """

    df = pd.read_json("evalset.jsonl", lines=True)

    # run unit tests on each column
    for idx, row in df.iterrows():
        query = row["query"]
        response = row["response"]  # In practice, you would sample your LLM or agent to get a response.

        # LMUnit tests are defined as natural language queries that are sent to the LMUnit model.
        # In this dataset we have multiple unit tests associated with a query.
        # You may think of this like a suite of tests that apply to each query and response pair. 
        unit_tests = row["unit_tests"]

        for unit_test in unit_tests:
            unit_test_score = lmunit(query, response, unit_test)
            assert unit_test_score >= 3.0, f"Failed unit test: \"{unit_test}\" for query: \"{query}\""
            time.sleep(1)  # wait to avoid rate limiting in the API


if __name__ == "__main__":
    test_evalset()
