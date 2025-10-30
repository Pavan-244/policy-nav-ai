# 🚀 Deployment Guide for PolicyNav on Render

## Prerequisites
- GitHub account
- Render account (free tier available at https://render.com)
- Git installed on your machine

## Step-by-Step Deployment Process

### 1. Prepare Your Code for GitHub

#### A. Initialize Git (if not already done)
```bash
cd Policy_Search_App
git init
```

#### B. Add all files
```bash
git add .
```

#### C. Commit your changes
```bash
git commit -m "Initial commit: PolicyNav AI application"
```

### 2. Push to GitHub

#### A. Create a new repository on GitHub
1. Go to https://github.com/new
2. Name: `policy-nav-ai` (or your preferred name)
3. Description: "AI-powered public policy search application"
4. Keep it Public or Private (both work with Render)
5. Don't initialize with README (we already have one)
6. Click "Create repository"

#### B. Link your local repo to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/policy-nav-ai.git
git branch -M main
git push -u origin main
```

### 3. Deploy on Render

#### Option A: Using Render Dashboard (Recommended)

1. **Sign up/Login to Render**
   - Go to https://dashboard.render.com/
   - Sign up with GitHub (recommended for easier connection)

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"

3. **Connect Repository**
   - Click "Connect account" if not connected
   - Select your `policy-nav-ai` repository
   - Click "Connect"

4. **Configure Web Service**
   ```
   Name:           policy-nav-ai
   Region:         Oregon (US West) or nearest to you
   Branch:         main
   Root Directory: (leave blank or set to Policy_Search_App if needed)
   Runtime:        Python 3
   Build Command:  pip install -r requirements.txt
   Start Command:  uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```

5. **Select Plan**
   - Choose "Free" plan (perfect for testing)
   - Free tier includes:
     - 750 hours/month
     - Auto-sleep after 15 min inactivity
     - Public URL

6. **Advanced Settings** (Optional)
   - Add environment variables if needed
   - Set Auto-Deploy: Yes (recommended)

7. **Create Web Service**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

#### Option B: Using render.yaml (Blueprint)

1. Go to Render Dashboard
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will detect `render.yaml` automatically
5. Click "Apply"

### 4. Post-Deployment

#### A. Check Deployment Status
- Monitor the build logs in Render dashboard
- Look for "Build successful" and "Deploy live"

#### B. Access Your Application
- Your app will be live at: `https://policy-nav-ai.onrender.com`
- Or the custom URL Render provides

#### C. Test All Features
- ✅ Home page loads
- ✅ Health search works
- ✅ Education search works
- ✅ Financial search works
- ✅ Quantum search works
- ✅ Visualization page works

### 5. Common Issues & Solutions

#### Issue: Build Fails
**Solution:**
- Check `requirements.txt` has all dependencies
- Verify Python version in `runtime.txt` is supported
- Check build logs for specific errors

#### Issue: App Crashes on Startup
**Solution:**
- Verify `Start Command` is correct
- Check that file paths are relative (not Windows absolute paths)
- Review application logs in Render dashboard

#### Issue: Static Files Not Loading
**Solution:**
- Ensure static files are in `frontend/static/`
- Check FastAPI static mount in `main.py`
- Verify file paths use `url_for('static', ...)`

#### Issue: 404 Errors for Routes
**Solution:**
- Check all routes in `backend/main.py`
- Verify template paths are correct
- Test routes locally before deploying

### 6. Updating Your Deployment

When you make changes:

```bash
# Make your changes
git add .
git commit -m "Description of changes"
git push origin main
```

Render will automatically:
- Detect the push
- Rebuild the application
- Deploy the new version
- Zero-downtime deployment

### 7. Custom Domain (Optional)

1. Go to your service in Render
2. Click "Settings" → "Custom Domain"
3. Add your domain
4. Update DNS records as instructed
5. Render provides free SSL certificate

### 8. Environment Variables

If you need to add secrets or config:

1. Render Dashboard → Your Service → Environment
2. Add variables:
   ```
   KEY=value
   ```
3. Access in code:
   ```python
   import os
   secret = os.getenv('KEY')
   ```

### 9. Monitoring & Logs

- **Logs**: Dashboard → Logs (real-time)
- **Metrics**: Dashboard → Metrics (CPU, Memory, Requests)
- **Alerts**: Set up in Dashboard → Settings

### 10. Cost Optimization

**Free Tier Limits:**
- 750 hours/month (enough for 24/7 uptime)
- 512 MB RAM
- 0.1 CPU
- Auto-sleep after 15 min inactivity

**To Upgrade:**
- Starter: $7/month (always on, no sleep)
- Standard: $25/month (more resources)

## Quick Reference Commands

```bash
# Check deployment status
curl https://policy-nav-ai.onrender.com

# View logs (if using Render CLI)
render logs

# Redeploy manually
# Go to Dashboard → Manual Deploy → Deploy Latest Commit
```

## Support & Troubleshooting

- **Render Docs**: https://render.com/docs
- **Render Status**: https://status.render.com
- **Community**: https://community.render.com

## Checklist Before Going Live

- [ ] All features tested locally
- [ ] requirements.txt is complete
- [ ] .gitignore excludes sensitive files
- [ ] README.md is up to date
- [ ] No hardcoded secrets in code
- [ ] Error pages are user-friendly
- [ ] Favicon is present
- [ ] HTTPS is enabled (automatic on Render)
- [ ] Custom domain configured (if desired)
- [ ] Monitoring is set up

## Your Live URLs

After deployment, update these in your README:

- **Application**: https://policy-nav-ai.onrender.com
- **Health Search**: https://policy-nav-ai.onrender.com/health
- **Education Search**: https://policy-nav-ai.onrender.com/education
- **Financial Search**: https://policy-nav-ai.onrender.com/financial
- **Quantum Search**: https://policy-nav-ai.onrender.com/quantum
- **Visualizations**: https://policy-nav-ai.onrender.com/visualize/nlp1

---

**🎉 Congratulations! Your PolicyNav AI application is now live on Render!**
