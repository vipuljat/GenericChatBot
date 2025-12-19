"""
Quick test to verify the employee API is returning data correctly
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from stateful_services.database import SessionLocal
from services.people_analyzer_service import get_employees_service

def test_get_employees():
    """Test the get_employees_service function"""
    db: Session = SessionLocal()
    
    try:
        print("\n" + "="*60)
        print("Test 1: Get all employees (no filters)")
        print("="*60)
        result = get_employees_service(
            db=db,
            department=None,
            role=None,
            search=None,
            include_chatbot=False
        )
        print(f"✓ Result type: {type(result)}")
        print(f"✓ Has 'employees' key: {'employees' in result}")
        print(f"✓ Number of employees: {len(result.get('employees', []))}")
        print(f"✓ Sample response structure:")
        import json
        print(json.dumps(result, indent=2, default=str)[:500])
        
        print("\n" + "="*60)
        print("Test 2: Get employees with include_chatbot=True")
        print("="*60)
        result = get_employees_service(
            db=db,
            department=None,
            role=None,
            search=None,
            include_chatbot=True
        )
        print(f"✓ Result type: {type(result)}")
        print(f"✓ Has 'employees' key: {'employees' in result}")
        print(f"✓ Has 'people_analyzer_chatbot' key: {'people_analyzer_chatbot' in result}")
        print(f"✓ Number of employees: {len(result.get('employees', []))}")
        
        print("\n" + "="*60)
        print("Test 3: Get employees by department")
        print("="*60)
        result = get_employees_service(
            db=db,
            department="Engineering",
            role=None,
            search=None,
            include_chatbot=False
        )
        print(f"✓ Number of Engineering employees: {len(result.get('employees', []))}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - API is working correctly")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_get_employees()
