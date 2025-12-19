# People Analyzer Chatbot Integration

## Overview
This document describes the integration of the People Analyzer chatbot with the employee list GET API endpoint.

## Changes Made

### 1. Service Layer (`services/people_analyzer_service.py`)

#### New Function: `get_people_analyzer_chatbot_service()`
```python
def get_people_analyzer_chatbot_service(db: Session, department: Optional[str] = None)
```

**Purpose**: Retrieves active people analyzer chatbot information, optionally filtered by department.

**Features**:
- Fetches chatbots with `mode = "people_analyzer"` and `status = "active"`
- Supports department-specific chatbots via metadata filtering
- Falls back to general chatbots if no department-specific chatbot exists
- Returns chatbot details including question count

**Returns**:
```json
{
  "chatbot_id": "uuid",
  "chatbot_name": "string",
  "description": "string",
  "instruction": "string",
  "question_count": 0,
  "mode": "people_analyzer",
  "status": "active"
}
```

#### Modified Function: `get_employees_service()`
**New Parameter**: `include_chatbot: bool = False`

**Behavior**:
- When `include_chatbot=True`, the response includes a `people_analyzer_chatbot` field
- The chatbot is fetched based on the provided department filter
- If no chatbot is found, the field is set to `null`

### 2. Routes Layer (`routes/people_analyzer_routes.py`)

#### Modified Endpoint: `GET /api/people-analyzer/employees`

**New Query Parameter**: `include_chatbot` (boolean, default: False)

**Updated Documentation**:
```
Parameters:
- department: Filter by department
- role: Filter by role
- search: Search by name or email
- include_chatbot: If True, includes people analyzer chatbot info for the department
```

## API Usage Examples

### Example 1: Get employees without chatbot info
```bash
GET /api/people-analyzer/employees?department=Engineering
```

**Response**:
```json
{
  "employees": [
    {
      "id": "uuid",
      "employee_id": "EMP001",
      "employee_name": "John Doe",
      "employee_email": "john@example.com",
      "department": "Engineering",
      "role": "Senior Engineer"
    }
  ]
}
```

### Example 2: Get employees WITH chatbot info
```bash
GET /api/people-analyzer/employees?department=Engineering&include_chatbot=true
```

**Response**:
```json
{
  "employees": [
    {
      "id": "uuid",
      "employee_id": "EMP001",
      "employee_name": "John Doe",
      "employee_email": "john@example.com",
      "department": "Engineering",
      "role": "Senior Engineer"
    }
  ],
  "people_analyzer_chatbot": {
    "chatbot_id": "uuid",
    "chatbot_name": "Engineering Performance Review",
    "description": "Performance review chatbot for engineering team",
    "instruction": "Answer questions about employee performance",
    "question_count": 15,
    "mode": "people_analyzer",
    "status": "active"
  }
}
```

### Example 3: No chatbot available
```bash
GET /api/people-analyzer/employees?department=HR&include_chatbot=true
```

**Response**:
```json
{
  "employees": [...],
  "people_analyzer_chatbot": null
}
```

## Chatbot Selection Logic

1. **Department-Specific Chatbot**: If a department parameter is provided, the system first looks for a chatbot with matching department metadata
   ```sql
   WHERE mode = 'people_analyzer' 
   AND status = 'active' 
   AND meta_data @> '{"department": "Engineering"}'
   ```

2. **General Chatbot**: If no department-specific chatbot exists, it falls back to a general people analyzer chatbot (one without department metadata)
   ```sql
   WHERE mode = 'people_analyzer' 
   AND status = 'active' 
   AND (meta_data IS NULL OR meta_data ? 'department' IS FALSE)
   ```

3. **No Chatbot**: Returns `null` if no chatbot matches the criteria

## Database Schema Requirements

### Chatbot Table
```sql
CREATE TABLE chatbots (
  chatbot_id UUID PRIMARY KEY,
  chatbot_name VARCHAR NOT NULL,
  description TEXT,
  instruction TEXT,
  mode VARCHAR DEFAULT 'general',
  status VARCHAR DEFAULT 'draft',
  meta_data JSONB,
  ...
);
```

**Key Fields**:
- `mode`: Must be set to `"people_analyzer"` for people analyzer chatbots
- `status`: Must be `"active"` for the chatbot to be included
- `meta_data`: Can contain `{"department": "DepartmentName"}` for department-specific chatbots

### Question Table
```sql
CREATE TABLE questions (
  question_id UUID PRIMARY KEY,
  chatbot_id UUID REFERENCES chatbots(chatbot_id),
  status VARCHAR DEFAULT 'active',
  question_data JSONB NOT NULL,
  ...
);
```

## Frontend Integration Guide

### Using the New Feature

```javascript
// Fetch employees with chatbot information
async function fetchEmployeesWithChatbot(department) {
  const response = await fetch(
    `/api/people-analyzer/employees?department=${department}&include_chatbot=true`,
    {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    }
  );
  
  const data = await response.json();
  
  // Check if chatbot is available
  if (data.people_analyzer_chatbot) {
    // Display chatbot option to user
    const chatbot = data.people_analyzer_chatbot;
    console.log(`Chatbot available: ${chatbot.chatbot_name}`);
    console.log(`Questions: ${chatbot.question_count}`);
    
    // Show "Start Review" button or similar UI
    showReviewButton(chatbot);
  } else {
    console.log('No chatbot available for this department');
    hideReviewButton();
  }
  
  // Display employees
  displayEmployees(data.employees);
}
```

## Testing

A test script is provided: `test_people_analyzer_integration.py`

**Run tests**:
```bash
# Make sure the backend is running
python test_people_analyzer_integration.py
```

**Test cases**:
1. Employee list without chatbot integration
2. Employee list with chatbot integration (specific department)
3. All departments with chatbot integration
4. Employee search with chatbot integration

## Benefits

1. **Single API Call**: Frontend can fetch both employees and chatbot information in one request
2. **Performance**: Optional parameter means no overhead when chatbot info isn't needed
3. **Flexibility**: Supports department-specific and general chatbots
4. **Scalability**: Easy to extend with additional filters or metadata

## Backward Compatibility

✅ **Fully backward compatible**
- Existing API calls continue to work without changes
- New parameter `include_chatbot` defaults to `False`
- Response structure remains the same when `include_chatbot=False`

## Future Enhancements

Possible improvements:
1. Cache chatbot lookups for better performance
2. Support multiple chatbots per department
3. Add chatbot assignment rules based on employee role
4. Include review completion status for each employee
5. Add chatbot preview/description expansion endpoints
