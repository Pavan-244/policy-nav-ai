# Git Setup and Deployment Script for PolicyNav
# Run this in PowerShell from the Policy_Search_App directory

Write-Host "🚀 PolicyNav - Git Setup and Deployment Helper" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if git is installed
Write-Host "Step 1: Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✓ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed. Please install Git from https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# Step 2: Initialize Git repository (if needed)
Write-Host ""
Write-Host "Step 2: Initializing Git repository..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
} else {
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
}

# Step 3: Configure Git (if needed)
Write-Host ""
Write-Host "Step 3: Checking Git configuration..." -ForegroundColor Yellow
$userName = git config user.name
$userEmail = git config user.email

if (-not $userName -or -not $userEmail) {
    Write-Host "Please configure Git with your details:" -ForegroundColor Cyan
    $name = Read-Host "Enter your name"
    $email = Read-Host "Enter your email"
    git config user.name "$name"
    git config user.email "$email"
    Write-Host "✓ Git configured successfully" -ForegroundColor Green
} else {
    Write-Host "✓ Git already configured" -ForegroundColor Green
    Write-Host "  Name: $userName" -ForegroundColor Gray
    Write-Host "  Email: $userEmail" -ForegroundColor Gray
}

# Step 4: Add files to Git
Write-Host ""
Write-Host "Step 4: Adding files to Git..." -ForegroundColor Yellow
git add .
Write-Host "✓ Files staged for commit" -ForegroundColor Green

# Step 5: Create initial commit
Write-Host ""
Write-Host "Step 5: Creating initial commit..." -ForegroundColor Yellow
try {
    git commit -m "Initial commit: PolicyNav AI application ready for deployment"
    Write-Host "✓ Initial commit created" -ForegroundColor Green
} catch {
    Write-Host "⚠ Commit may already exist or no changes to commit" -ForegroundColor Yellow
}

# Step 6: Instructions for GitHub
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Next Steps - Create GitHub Repository:" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to: https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: policy-nav-ai" -ForegroundColor White
Write-Host "3. Description: AI-powered public policy search application" -ForegroundColor White
Write-Host "4. Keep it Public" -ForegroundColor White
Write-Host "5. DON'T initialize with README" -ForegroundColor White
Write-Host "6. Click 'Create repository'" -ForegroundColor White
Write-Host ""
Write-Host "After creating the repository, run these commands:" -ForegroundColor Cyan
Write-Host ""
Write-Host "git remote add origin https://github.com/YOUR_USERNAME/policy-nav-ai.git" -ForegroundColor Yellow
Write-Host "git branch -M main" -ForegroundColor Yellow
Write-Host "git push -u origin main" -ForegroundColor Yellow
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Deploy to Render:" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to: https://dashboard.render.com/" -ForegroundColor White
Write-Host "2. Click 'New +' → 'Web Service'" -ForegroundColor White
Write-Host "3. Connect your GitHub repository" -ForegroundColor White
Write-Host "4. Use these settings:" -ForegroundColor White
Write-Host "   - Build Command: pip install -r requirements.txt" -ForegroundColor Gray
Write-Host "   - Start Command: uvicorn backend.main:app --host 0.0.0.0 --port `$PORT" -ForegroundColor Gray
Write-Host "5. Click 'Create Web Service'" -ForegroundColor White
Write-Host ""
Write-Host "📚 For detailed instructions, see DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "✨ Setup complete! Your code is ready for deployment." -ForegroundColor Green
