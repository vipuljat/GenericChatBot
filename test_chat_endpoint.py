"""
Test script for RAG chat endpoint
Tests the /chatbots/{id}/chat endpoint with a sample query
"""
import requests
import json

API_URL = "http://localhost:8000/chatbot/chatbots"
CHATBOT_ID = 1  # Use the chatbot ID from previous upload test

# Test query
test_query = "What is this document about?"

print(f"Testing RAG chat with chatbot {CHATBOT_ID}")
print(f"Query: {test_query}\n")

# Make request to chat endpoint
chat_url = f"{API_URL}/{CHATBOT_ID}/chat"
payload = {
    "query": test_query,
    "conversation_history": None
}

print(f"Sending request to: {chat_url}")
response = requests.post(chat_url, json=payload)

print(f"\nStatus code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print("\n" + "="*60)
    print("RESPONSE:")
    print("="*60)
    print(result['response'])
    print("\n" + "="*60)
    print(f"SOURCES ({len(result['sources'])} documents used):")
    print("="*60)
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['doc_name']} (Relevance: {source['relevance_score']:.3f})")
    
    print("\n" + "="*60)
    print(f"Context used: {result['context_used']}")
    print(f"Number of chunks: {result.get('num_chunks_used', 0)}")
    print("="*60)
else:
    print(f"\n❌ Error: {response.text}")

print("\n\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nTry different queries:")
print("- 'What are the main topics covered?'")
print("- 'Can you summarize the key points?'")
print("- 'How many leaves am I entitled to?' (if HR document)")
