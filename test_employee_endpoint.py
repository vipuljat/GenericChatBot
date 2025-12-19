"""
Simple script to test the employee list endpoint directly
Run this to verify the API is returning data correctly
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# You'll need to update this with a valid token
AUTH_TOKEN = "YOUR_AUTH_TOKEN_HERE"

def test_employees_endpoint():
    """Test the employees endpoint"""
    
    print("\n" + "="*70)
    print("TESTING EMPLOYEE LIST ENDPOINT")
    print("="*70)
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Get all employees
    print("\n📋 Test 1: Get all employees (no filters)")
    print("-" * 70)
    try:
        response = requests.get(
            f"{BASE_URL}/api/people-analyzer/employees",
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS")
            print(f"   - Has 'employees' key: {'employees' in data}")
            print(f"   - Number of employees: {len(data.get('employees', []))}")
            print(f"   - Response structure: {list(data.keys())}")
            if data.get('employees'):
                print(f"   - First employee: {json.dumps(data['employees'][0], indent=6)}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Test 2: Get employees with include_chatbot=true
    print("\n📋 Test 2: Get employees WITH chatbot info")
    print("-" * 70)
    try:
        response = requests.get(
            f"{BASE_URL}/api/people-analyzer/employees",
            headers=headers,
            params={"include_chatbot": "true"},
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS")
            print(f"   - Has 'employees' key: {'employees' in data}")
            print(f"   - Has 'people_analyzer_chatbot' key: {'people_analyzer_chatbot' in data}")
            print(f"   - Number of employees: {len(data.get('employees', []))}")
            if data.get('people_analyzer_chatbot'):
                print(f"   - Chatbot: {data['people_analyzer_chatbot'].get('chatbot_name')}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Test 3: Get employees by department
    print("\n📋 Test 3: Get employees by department")
    print("-" * 70)
    try:
        response = requests.get(
            f"{BASE_URL}/api/people-analyzer/employees",
            headers=headers,
            params={"department": "Engineering"},
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS")
            print(f"   - Number of Engineering employees: {len(data.get('employees', []))}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETED")
    print("="*70)
    print("\n💡 Note: Update AUTH_TOKEN in this script with a valid token")
    print("💡 Make sure the backend server is running on http://localhost:8000\n")


if __name__ == "__main__":
    if AUTH_TOKEN == "YOUR_AUTH_TOKEN_HERE":
        print("\n⚠️  WARNING: Please update AUTH_TOKEN in this script first!")
        print("   You can get a token by logging in through the API\n")
    
    test_employees_endpoint()
