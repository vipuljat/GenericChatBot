"""
End-to-end test: Upload HR document and test RAG queries
"""
import requests
import time
import os

API_URL = "http://localhost:8000/chatbot/chatbots"

# Step 1: Create HR policy document
print("Creating sample HR policy document...")
hr_document = """
COMPANY HR POLICY DOCUMENT
===========================

LEAVE POLICY
------------
All full-time employees are entitled to the following leave benefits:

1. Annual Leave: 25 days per calendar year
2. Sick Leave: 15 days per calendar year
3. Casual Leave: 10 days per calendar year
4. Maternity Leave: 180 days (6 months) for female employees
5. Paternity Leave: 15 days for male employees

Leave can be carried forward up to 5 days to the next year. 
Unused leave beyond this will be forfeited.

WORKING HOURS
-------------
Standard working hours are 9:00 AM to 6:00 PM, Monday to Friday.
Employees are expected to work 8 hours per day excluding lunch break.
Lunch break duration: 1 hour (flexible between 12 PM and 2 PM)

REMOTE WORK POLICY
------------------
Employees can work remotely up to 2 days per week with manager approval.
Full remote work requires VP-level approval and is subject to role suitability.

COMPENSATION AND BENEFITS
-------------------------
- Annual salary reviews in January
- Performance bonus: up to 20% of annual salary based on performance rating
- Health insurance: Coverage for employee, spouse, and up to 2 children
- Provident Fund: Company contributes 12% of basic salary
- Gratuity: Applicable after 5 years of continuous service

PROBATION PERIOD
----------------
New employees serve a probation period of 6 months.
During probation, notice period is 1 month from either side.
After confirmation, notice period is 3 months from either side.

CODE OF CONDUCT
---------------
All employees must:
- Maintain confidentiality of company information
- Respect diversity and practice inclusion
- Report any harassment or discrimination immediately
- Avoid conflicts of interest
- Follow cybersecurity protocols when handling company data
"""

# Save document
doc_path = "hr_policy_document.txt"
with open(doc_path, "w", encoding="utf-8") as f:
    f.write(hr_document)

print("✅ HR document created\n")

# Step 2: Upload and create HR chatbot
print("Uploading HR document and creating chatbot...")
form_data = {
    "chatbot_name": (None, "HR Assistant"),
    "description": (None, "HR chatbot to help employees with HR policies and queries"),
    "instructions": (None, "You are an HR assistant. Answer employee questions about company policies, leave entitlements, benefits, and other HR-related topics. Be professional, accurate, and cite the relevant policy sections."),
    "is_quiz_mode": (None, "false"),
    "is_active": (None, "true"),
    "recommended_model": (None, "gemini-2.0-flash-exp"),
}

with open(doc_path, "rb") as f:
    files = {"file": (os.path.basename(doc_path), f, "text/plain")}
    response = requests.post(API_URL, files=files, data=form_data)

if response.status_code != 200:
    print(f"❌ Failed to create chatbot: {response.text}")
    os.remove(doc_path)
    exit(1)

result = response.json()
chatbot_id = result["chatbot_id"]
print(f"✅ HR chatbot created with ID: {chatbot_id}")

# Wait for embeddings
print("⏳ Waiting for embeddings to be generated...")
time.sleep(5)

# Step 3: Test various HR queries
test_queries = [
    "How many leaves am I entitled to?",
    "What is the annual leave policy?",
    "Can I work from home?",
    "What are the working hours?",
    "How much is the performance bonus?",
    "What is the notice period during probation?",
    "Do I get health insurance for my family?",
]

print("\n" + "=" * 70)
print("TESTING RAG CHAT ENDPOINT")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n[Query {i}/{len(test_queries)}]: {query}")
    print("-" * 70)

    chat_url = f"{API_URL}/{chatbot_id}/chat"
    payload = {"query": query}

    response = requests.post(chat_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Response:\n{result['response']}\n")
        print(f"📚 Sources: {len(result['sources'])} document chunks used")

        # FIXED: Safe & readable relevance score list
        scores = [f"{s['relevance_score']:.3f}" for s in result["sources"]]
        print(f"   Relevance scores: {scores}")

    else:
        print(f"❌ Error: {response.text}")

    print()
    time.sleep(1)  # prevent rate limit issues

# Cleanup
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\n✅ All tests completed!")
print(f"📝 HR Chatbot ID: {chatbot_id}")
print(f"🔗 Try it yourself: POST http://localhost:8000/chatbot/chatbots/{chatbot_id}/chat")
print(f'   Body: {{"query": "your question here"}}')

# Clean up test document
os.remove(doc_path)
print("\n🧹 Cleaned up test document")
