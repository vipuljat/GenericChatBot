# Start Backend Server

Write-Host "🚀 Starting ChatBot Backend Server..." -ForegroundColor Cyan
Write-Host ""

# Check if we're in the correct directory
if (-not (Test-Path ".\main.py")) {
    Write-Host "❌ Error: Please run this script from the ChatBot directory" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path ".\myvenv")) {
    Write-Host "❌ Error: Virtual environment not found" -ForegroundColor Red
    Write-Host "   Please create a virtual environment first" -ForegroundColor Yellow
    exit 1
}

Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& ".\myvenv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "📝 Checking configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".\.env")) {
    Write-Host "⚠️  Warning: .env file not found" -ForegroundColor Yellow
    Write-Host "   Make sure to configure environment variables" -ForegroundColor White
}

Write-Host ""
Write-Host "🌐 Starting Uvicorn server..." -ForegroundColor Cyan
Write-Host "   Backend will be available at: http://localhost:8000" -ForegroundColor White
Write-Host "   API docs at: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

uvicorn main:app --reload
