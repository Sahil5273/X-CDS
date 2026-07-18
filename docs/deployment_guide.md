# Step-by-Step Deployment Guide: Google Cloud Run & Firebase Hosting

This guide outlines how to deploy the **X-CDS** application to a live, production-grade cloud environment using **Google Cloud Run** for the FastAPI backend and **Firebase Hosting** for the React frontend.

---

## Prerequisites
1.  **Google Cloud SDK (gcloud CLI)** installed on your machine.
2.  **Node.js & npm** installed.
3.  **Firebase CLI** installed (`npm install -g firebase-tools`).

---

## Part 1: Deploying the Backend to Google Cloud Run

Google Cloud Run builds and runs your containerized FastAPI backend automatically from your source files.

### Step 1: Initialize gcloud and Set Project
Open your terminal and run:
```bash
# Log in to your Google Cloud account
gcloud auth login

# Set your active project ID
gcloud config set project sahil-portfolio-76e8d
```

### Step 2: Enable GCP APIs
Enable the necessary services in your Google Cloud console:
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

### Step 3: Deploy the Container
Run the deploy command from your **project root directory** (where your backend files are located):
```bash
gcloud run deploy xcds-backend --source . --region us-central1 --allow-unauthenticated
```
*   *Note:* The deployment command will build your Docker image using Google Cloud Build and host it.
*   Once finished, it will output a **Service URL** (e.g., `https://xcds-backend-76e8d-uc.a.run.app`). Copy this URL.

---

## Part 2: Configuring and Deploying the React Frontend

Now, we will point your React app to the live backend URL and upload it to Firebase Hosting.

### Step 1: Update the Frontend API URL
Create or update your frontend environment variables so the React client calls your live Cloud Run API instead of `localhost:8000`.

In your frontend folder, update/create **`.env.production`**:
```ini
VITE_API_BASE_URL=https://xcds-backend-76e8d-uc.a.run.app
```

### Step 2: Build the React Application
In your frontend directory, compile the production static files:
```bash
npm run build
```
This compiles your website into a static folder named **`dist`** (or `build`).

### Step 3: Initialize Firebase Hosting
Log in to Firebase and link the project:
```bash
firebase login
```
Then run the hosting setup:
```bash
firebase init hosting
```
During initialization, select:
1.  **Project:** Select `Use an existing project` and choose `sahil-portfolio-76e8d`.
2.  **Public Directory:** Type **`dist`** (Vite's build output folder).
3.  **Configure as single-page app:** Type **`Yes`**.
4.  **Set up automatic builds with GitHub:** Type **`No`** (unless you want CI/CD).

### Step 4: Deploy the Frontend to Firebase
Run the deploy command to publish your webapp live:
```bash
firebase deploy --only hosting
```
Firebase will output a live hosting URL (e.g., `https://sahil-portfolio-76e8d.web.app`).

---

## Part 3: Verify the Live Deployment
Open your live Firebase URL in a web browser. Test a clinical query to verify that:
1.  The frontend loads instantly.
2.  FastAPI on Cloud Run processes the query.
3.  Vertex AI computes responses and returns verified citations!
