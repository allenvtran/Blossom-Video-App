# Blossom-Video-App

A simple Streamlit application for video content.

## 🚀 Deploy on Streamlit Cloud

To deploy this app on Streamlit Cloud, follow these steps:

### Prerequisites
- A GitHub account
- This repository pushed to GitHub

### Deployment Steps

1. **Visit Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy Your App**
   - Click "New app" button
   - Select your repository: `allenvtran/Blossom-Video-App`
   - Choose the branch (usually `main` or `master`)
   - Set the main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Streamlit will automatically install dependencies from `requirements.txt`
   - Your app will be live at: `https://[your-app-name].streamlit.app`

### Alternative: Deploy from GitHub

You can also deploy directly from this repository by clicking the button below:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 🏃 Run Locally

To run this app on your local machine:

```bash
# Clone the repository
git clone https://github.com/allenvtran/Blossom-Video-App.git
cd Blossom-Video-App

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
Blossom-Video-App/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml    # Streamlit configuration
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## 🛠️ Requirements

- Python 3.7+
- Streamlit 1.28.0 or higher

## 📝 Notes

- Make sure your repository is public or you have granted Streamlit access to private repositories
- The app will automatically restart when you push changes to your GitHub repository
- Logs and errors can be viewed in the Streamlit Cloud dashboard

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available for educational purposes.