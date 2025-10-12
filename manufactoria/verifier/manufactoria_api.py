"""
Manufactoria DSL verification API.

Can launch local server with:
```
uv run nohup uvicorn manufactoria.eval.manufactoria_api:app --host 0.0.0.0 --port 1235 &
```

or launch the server in a docker container:
```
docker build -t manufactoria-api -f open_instruct/code/Dockerfile.manufactoria .
docker run -p 1235:1235 manufactoria-api
```

and then test with:
```
python open_instruct/code/manufactoria_api.py
```

or

```
curl -X GET http://localhost:1235/health
curl -X POST http://localhost:1235/test_solution -H "Content-Type: application/json" -d '{
    "dsl": "START start:\n    NEXT end\nEND end", 
    "test_cases": [{"input": "", "expected_output": "", "expected_accepted": true, "check_output": false}],
    "max_execution_time": 1.0
}'
```
"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .manufactoria_parser import create_robot_factory, ParseError
import re

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCase(BaseModel):
    input: str
    expected_output: str = ""
    expected_accepted: bool = True
    check_output: bool = True
    description: str = ""


class ManufactoriaTestRequest(BaseModel):
    dsl: str
    test_cases: List[TestCase]
    max_execution_time: float = 1.0


@app.post("/test_solution")
async def test_solution(request: ManufactoriaTestRequest):
    """Test a Manufactoria DSL solution against test cases."""
    try:
        # Create factory from DSL
        factory = create_robot_factory(request.dsl)
        results = []
        
        for test_case in request.test_cases:
            result = factory.process_robot(test_case.input)
            
            if test_case.check_output:
                # Check if expected_output contains regex patterns
                has_regex_patterns = any(char in test_case.expected_output for char in ['.', '+', '*', '?', '|', '(', ')'])
                
                if has_regex_patterns:
                    # Use regex matching
                    try:
                        output_matches = bool(re.fullmatch(test_case.expected_output, result.final_tape))
                    except re.error:
                        # If regex pattern is invalid, fall back to exact matching
                        output_matches = result.final_tape == test_case.expected_output
                else:
                    # Use exact matching
                    output_matches = result.final_tape == test_case.expected_output
                
                passed = (output_matches and result.finished) == test_case.expected_accepted
            else:
                passed = (result.finished == test_case.expected_accepted)
            
            test_result = {
                'input': test_case.input,
                'expected_output': test_case.expected_output,
                'actual_output': result.final_tape,
                'expected_accepted': test_case.expected_accepted,
                'actual_accepted': result.finished,
                'check_output': test_case.check_output,
                'passed': passed,
                'path': result.path,
                'rejection_reason': result.rejection_reason,
                'description': test_case.description
            }
            results.append(test_result)
        
        all_passed = all(r['passed'] for r in results)
        
        return {
            "valid": True,
            "all_passed": all_passed,
            "results": results
        }
        
    except ParseError as e:
        return {
            "valid": False,
            "all_passed": False,
            "message": f"DSL Parse Error: {str(e)}",
            "results": []
        }
    except Exception as e:
        return {
            "valid": False,
            "all_passed": False,
            "message": f"Unexpected error: {str(e)}",
            "results": []
        }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import requests

    # API endpoint
    url = "http://localhost:1235/test_solution"

    # Test data - simple DSL that just accepts empty input
    payload = {
        "dsl": """
START start:
    NEXT end

END end
""",
        "test_cases": [
            {
                "input": "",
                "expected_output": "",
                "expected_accepted": True,
                "check_output": False,
                "description": "Empty input should be accepted"
            },
            {
                "input": "R",
                "expected_output": "",
                "expected_accepted": False,
                "check_output": False,
                "description": "Non-empty input should be rejected (no path to end)"
            }
        ],
        "max_execution_time": 1.0,
    }

    # Send POST request
    response = requests.post(url, json=payload)

    response_json = response.json()
    # Print results
    print("Status Code:", response.status_code)
    print("Response:", response_json)

    # Check if all tests passed
    assert response_json["all_passed"] == True
    print("✓ All tests passed!")