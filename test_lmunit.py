import pandas as pd
import time
import os
from contextual import ContextualAI

# Set up Contextual SDK
contextual_api_key = os.environ.get("CONTEXTUAL_API_KEY")
client = ContextualAI(api_key = contextual_api_key, base_url = "https://api.contextual.ai/v1")


# Global unit tests apply to all queries in your evaluation set
global_unit_tests = ["Does the response directly address the prompt’s query or topic?"]


def test_evalset():
    """
    This method retrieves queries and unit tests from an evalset.jsonl file. 
    For each query, it generates a response using Contextual's Grounded Language Model. 
    Then, it uses LMUnit to evaluate each response against each relevant unit test. 
    Alternatively, you could use the @pytest.mark.parametrize method.
    """

    df = pd.read_json("evalset.jsonl", lines=True)

    # run unit tests for each row in the evalset
    for idx, row in df.iterrows():
        query = row["query"]
        knowledge = row["knowledge"]  # This is the knowledge that Contextual's Grounded Language Model will use to generate a response.
        response = client.generate.create(
            model="v1",
            messages=[{"role": "user", "content": query}],
            knowledge=[knowledge],      
        ).response

        # LMUnit tests are defined as natural language queries that are sent to the LMUnit model.
        # In this dataset we have multiple unit tests associated with a query.
        # You may think of this like a suite of tests that apply to each query and response pair. 
        unit_tests = row["unit_tests"] + global_unit_tests

        for unit_test in unit_tests:
            unit_test_score = client.lmunit.create(query=query, response=response, unit_test=unit_test).score
            assert unit_test_score >= 3.0, f"Failed unit test: \"{unit_test}\" for query: \"{query}\""  # score of 3.0 is passing
            time.sleep(1)  # wait to avoid rate limiting in the API


if __name__ == "__main__":
    test_evalset()
