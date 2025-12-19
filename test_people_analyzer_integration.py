"""
Test script for People Analyzer Chatbot Integration with Employee List API
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000/api/people-analyzer"

def test_employee_list_without_chatbot():
    """Test employee list endpoint without chatbot integration"""
    print("\n" + "="*60)
    print("Test 1: Employee List WITHOUT Chatbot Integration")
    print("="*60)
    
    url = f"{BASE_URL}/employees"
    params = {
        "department": "Engineering"
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nNumber of employees: {len(data.get('employees', []))}")
            print(f"Chatbot info included: {'people_analyzer_chatbot' in data}")
            print(f"\nResponse structure:")
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def test_employee_list_with_chatbot():
    """Test employee list endpoint WITH chatbot integration"""
    print("\n" + "="*60)
    print("Test 2: Employee List WITH Chatbot Integration")
    print("="*60)
    
    url = f"{BASE_URL}/employees"
    params = {
        "department": "Engineering",
        "include_chatbot": True
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nNumber of employees: {len(data.get('employees', []))}")
            print(f"Chatbot info included: {'people_analyzer_chatbot' in data}")
            
            if 'people_analyzer_chatbot' in data:
                chatbot_info = data['people_analyzer_chatbot']
                if chatbot_info:
                    print(f"\nPeople Analyzer Chatbot Details:")
                    print(f"  - Chatbot ID: {chatbot_info.get('chatbot_id')}")
                    print(f"  - Name: {chatbot_info.get('chatbot_name')}")
                    print(f"  - Description: {chatbot_info.get('description')}")
                    print(f"  - Question Count: {chatbot_info.get('question_count')}")
                    print(f"  - Mode: {chatbot_info.get('mode')}")
                    print(f"  - Status: {chatbot_info.get('status')}")
                else:
                    print("\nNo active people analyzer chatbot found for this department")
            
            print(f"\nFull Response:")
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def test_employee_list_all_departments_with_chatbot():
    """Test employee list endpoint for all departments WITH chatbot"""
    print("\n" + "="*60)
    print("Test 3: All Departments WITH Chatbot Integration")
    print("="*60)
    
    url = f"{BASE_URL}/employees"
    params = {
        "include_chatbot": True
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nTotal employees: {len(data.get('employees', []))}")
            print(f"Chatbot info included: {'people_analyzer_chatbot' in data}")
            
            if 'people_analyzer_chatbot' in data:
                chatbot_info = data['people_analyzer_chatbot']
                if chatbot_info:
                    print(f"\nGeneral People Analyzer Chatbot Found:")
                    print(f"  - Name: {chatbot_info.get('chatbot_name')}")
                    print(f"  - Question Count: {chatbot_info.get('question_count')}")
                else:
                    print("\nNo general people analyzer chatbot found")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def test_employee_search_with_chatbot():
    """Test employee search with chatbot integration"""
    print("\n" + "="*60)
    print("Test 4: Employee Search WITH Chatbot Integration")
    print("="*60)
    
    url = f"{BASE_URL}/employees"
    params = {
        "search": "John",
        "include_chatbot": True
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nSearch results: {len(data.get('employees', []))} employees")
            print(f"Chatbot info included: {'people_analyzer_chatbot' in data}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PEOPLE ANALYZER CHATBOT INTEGRATION TESTS")
    print("="*60)
    print("\nNote: Make sure the backend server is running on http://localhost:8000")
    print("Note: You need a valid authentication token to run these tests")
    print("\nThese tests will fail with 401 Unauthorized without proper authentication.")
    print("="*60)
    
    # Run tests
    test_employee_list_without_chatbot()
    test_employee_list_with_chatbot()
    test_employee_list_all_departments_with_chatbot()
    test_employee_search_with_chatbot()
    
    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)
