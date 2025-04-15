# VisionAID

VisionAID is an AI-powered assistive tool for the visually impaired. It converts images into spoken descriptions, detects surroundings, and guides navigation using voice, GPS, and Google Maps.

## Features

- Image captioning with AI-generated descriptions
- Live object detection with voice announcements
- Smart navigation with voice guidance
- Voice command interface
- Mobile-responsive design

## Deployment on Vercel

Follow these steps to deploy VisionAID on Vercel:

### Prerequisites

1. A [Vercel](https://vercel.com) account
2. [Vercel CLI](https://vercel.com/download) installed (optional for CLI deployment)
3. [Git](https://git-scm.com/downloads) installed

### Deployment Steps

#### Option 1: Deploy via GitHub

1. Push your code to a GitHub repository
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. In the Vercel dashboard:
   - Click "Add New" → "Project"
   - Select your GitHub repository
   - Configure your project settings (Vercel should auto-detect Python)
   - Click "Deploy"

#### Option 2: Deploy via Vercel CLI

1. Login to Vercel CLI
   ```bash
   vercel login
   ```

2. Deploy the project
   ```bash
   vercel
   ```

3. Follow the prompts to configure your deployment

### Environment Variables

Make sure to add these environment variables in your Vercel project settings:

- `GOOGLE_API_KEY`: Your Google Maps API key

### Notes for Vercel Deployment

- The `vercel.json` file configures the deployment settings
- Large ML models should be hosted separately and downloaded at runtime
- Create a directory structure that works with Vercel's serverless functions

## Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   python wsgi.py
   ```

## Project Structure

```
.
├── app.py               # Main Flask application
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel configuration
├── wsgi.py              # Development server 
├── static/              # Static files
└── templates/           # HTML templates
```

## License

Copyright © 2025 VisionAID Team 