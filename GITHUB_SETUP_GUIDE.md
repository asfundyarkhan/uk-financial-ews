# 🚀 GitHub Repository Setup Guide

**UK Financial Early Warning System - Final Year Project**

Follow this step-by-step guide to push your project to GitHub.

---

## ✅ Pre-Push Checklist

Before pushing to GitHub, verify:

- [x] README.md created with comprehensive documentation
- [x] .gitignore configured (excludes data files, outputs)
- [x] requirements.txt lists all dependencies
- [x] LICENSE file added (MIT License)
- [x] Documentation organized in `docs/` folder
- [x] Data directory created with instructions
- [x] Unnecessary files removed
- [x] Core code files present and functional

---

## 📋 Step-by-Step GitHub Push

### Step 1: Create GitHub Repository

1. **Go to GitHub**: https://github.com
2. **Sign in** to your account
3. Click **"+"** → **"New repository"**
4. Fill in details:
   - **Repository name**: `uk-financial-ews` (or your preferred name)
   - **Description**: "Machine Learning-based Early Warning System for UK Financial Market Stress Prediction - Final Year Project"
   - **Visibility**: ✅ Public (for academic access)
   - **DON'T** initialize with README, .gitignore, or license (we already have them)
5. Click **"Create repository"**

---

### Step 2: Prepare Local Repository

Open PowerShell in your project directory:

```powershell
cd "c:\Final year Project"
```

**Check Git status:**
```powershell
git status
```

If `.git` folder doesn't exist, initialize:
```powershell
git init
```

---

### Step 3: Stage Files for Commit

**Add all files (respecting .gitignore):**
```powershell
git add .
```

**Verify what will be committed:**
```powershell
git status
```

You should see:
- ✅ Python scripts (.py files)
- ✅ Documentation (README.md, LICENSE, etc.)
- ✅ Configuration files (.gitignore, requirements.txt)
- ❌ NOT CSV files (excluded by .gitignore)
- ❌ NOT large outputs (excluded by .gitignore)

---

### Step 4: Create Initial Commit

```powershell
git commit -m "Initial commit: UK Financial EWS - Final Year Project

- Complete ML pipeline for financial stress prediction
- XGBoost model with 93.26% ROC-AUC
- 103 engineered features from FTSE, Gold, Silver
- Comprehensive documentation and setup guides
- Academic project for [Your University Name]"
```

---

### Step 5: Connect to GitHub

**Add remote repository** (replace with YOUR GitHub username):

```powershell
git remote add origin https://github.com/YOUR_USERNAME/uk-financial-ews.git
```

**Verify remote:**
```powershell
git remote -v
```

---

### Step 6: Push to GitHub

**For first push:**
```powershell
git branch -M main
git push -u origin main
```

**Enter credentials when prompted** (use Personal Access Token, not password)

---

### Step 7: Verify Upload

1. Refresh your GitHub repository page
2. Confirm files are uploaded
3. Check README renders correctly

---

## 🔐 Authentication Options

### Option 1: Personal Access Token (Recommended)

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Select scopes: `repo` (full control)
4. Copy token (save it securely!)
5. Use token as password when git prompts

### Option 2: GitHub CLI

```powershell
# Install GitHub CLI
winget install GitHub.cli

# Authenticate
gh auth login

# Push
git push -u origin main
```

### Option 3: SSH Key

```powershell
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@university.ac.uk"

# Add to GitHub: Settings → SSH and GPG keys
```

---

## 📝 Update README with Your Details

Before pushing, personalize the README:

### Update These Sections:

1. **Line 329** - Your name
2. **Line 330** - Your email
3. **Line 331** - Your GitHub username
4. **Line 332** - Your LinkedIn
5. **Line 368** - Your university name
6. **Line 369** - Your degree program
7. **Line 371** - Your supervisor's name
8. **Top badges** - Update repository URL

**Edit README.md:**
```powershell
code README.md
# Or use any text editor
notepad README.md
```

**Then commit changes:**
```powershell
git add README.md
git commit -m "Update: Personalize README with author details"
git push
```

---

## 🎨 Customize Repository Settings

### On GitHub Repository Page:

1. **About Section** (right sidebar):
   - Click ⚙️ (gear icon)
   - Add description
   - Add topics: `machine-learning`, `financial-analysis`, `early-warning-system`, `xgboost`, `time-series`, `python`, `academic-project`
   - Add website (if you have project page)

2. **README.md Link** - Update repository URL in badges

3. **GitHub Pages** (optional):
   - Settings → Pages
   - Deploy from branch: `main` → `/docs`
   - Creates: `https://your-username.github.io/uk-financial-ews/`

---

## 📊 Repository Structure on GitHub

Your repository will look like:

```
uk-financial-ews/
├── 📄 README.md (renders on homepage)
├── 📄 LICENSE
├── 📄 CONTRIBUTING.md
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📄 .gitattributes
│
├── 📂 data/
│   └── README.md (data acquisition guide)
│
├── 📂 docs/
│   ├── FEATURE_EXTRACTION_README.md
│   └── PROJECT_SUMMARY.md
│
├── 📜 data_cleaning.py
├── 📜 complete_feature_extraction.py
├── 📜 financial_ews_model.py
└── 📜 financial_ews_analysis.py
```

**Note**: CSV files and outputs are NOT uploaded (excluded by .gitignore)

---

## 🔄 Future Updates

### Making Changes:

```powershell
# Make your code changes

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add: New feature description"

# Push to GitHub
git push
```

### Best Practices:

- **Commit often** with clear messages
- **Test before pushing**
- **Update documentation** when adding features
- **Use branches** for major changes:
  ```powershell
  git checkout -b feature/new-feature
  # Make changes
  git push -u origin feature/new-feature
  # Create Pull Request on GitHub
  ```

---

## 🎓 Adding GitHub Link to Your Report

### For Your Final Project Report (FPR):

**On the Title Page or Introduction:**

```
Project Repository: https://github.com/YOUR_USERNAME/uk-financial-ews
```

**Or use a QR code** (generates automatically on GitHub):
- Repository → Code → Download ZIP → Use QR code generator

**Or add a badge** in your document:

```
GitHub Repository: [View Code]
Available at: https://github.com/YOUR_USERNAME/uk-financial-ews
```

---

## ✅ Verification Checklist

After pushing to GitHub, verify:

- [ ] Repository is public and accessible
- [ ] README.md displays correctly with formatting
- [ ] All Python scripts are uploaded
- [ ] Documentation is in `docs/` folder
- [ ] Data README explains how to download datasets
- [ ] Large CSV files are NOT uploaded (check repo size)
- [ ] License is visible
- [ ] requirements.txt is complete
- [ ] Repository has descriptive topics/tags
- [ ] Your name and contact info are updated in README

---

## 🆘 Troubleshooting

### Problem: "Permission denied" when pushing
**Solution**: Use Personal Access Token instead of password

### Problem: Repository too large
**Solution**: Check if CSV files were accidentally committed:
```powershell
git rm --cached *.csv
git commit -m "Remove large CSV files"
git push
```

### Problem: README not rendering correctly
**Solution**: Check Markdown syntax, preview locally:
```powershell
# Install Markdown preview
code README.md
# Use Ctrl+Shift+V to preview
```

### Problem: Wrong files in repository
**Solution**: Update .gitignore and remove from Git:
```powershell
git rm --cached filename.ext
git commit -m "Remove unnecessary file"
git push
```

---

## 📧 Support

If you need help:
1. Check GitHub Docs: https://docs.github.com/
2. Ask your supervisor
3. Contact university IT support

---

## 🎉 You're Ready!

Your repository is now:
- ✅ Professionally structured
- ✅ Well-documented
- ✅ Academically rigorous
- ✅ Accessible to anyone
- ✅ Ready for your Final Project Report

**Good luck with your project submission! 🚀**

---

**Created**: April 2026  
**For**: Final Year Project - UK Financial Early Warning System
