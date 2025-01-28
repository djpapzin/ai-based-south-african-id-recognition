# Label Studio Setup Instructions

## Prerequisites
- Python 3.8 or higher installed on your system
- pip (Python package installer)
- A command prompt or terminal

## Setup Steps

1. **Create and activate a virtual environment**:
   ```powershell
   # Create a virtual environment
   python -m venv .venv

   # Activate the virtual environment (Windows PowerShell)
   .\.venv\Scripts\Activate
   ```

2. **Install Label Studio**:
   ```powershell
   pip install label-studio
   ```

3. **Start Label Studio**:
   ```powershell
   # Start the Label Studio server
   label-studio start --host localhost --port 8090
   ```

4. **Access Label Studio**:
   - Open your web browser
   - Go to: http://localhost:8090
   - Create an account on first launch

## Common Issues and Solutions

- If Label Studio fails to start, try:
  - Ensuring no other application is using port 8090
  - Using a different port (e.g., `--port 8095`)
  - Checking that Python and pip are properly installed

- If you can't access the web interface:
  - Verify the server is running in your terminal
  - Try accessing with `127.0.0.1` instead of `localhost`
  - Check if your firewall is blocking the connection

## Basic Usage

1. **Create a Project**:
   - Click "Create Project" after logging in
   - Give your project a name and description
   - Select the type of labeling you'll be doing

2. **Import Data**:
   - Go to your project
   - Click "Import" to add data
   - Choose your data source and format

3. **Start Labeling**:
   - Click "Label All Tasks" to begin
   - Follow the labeling instructions provided
   - Use keyboard shortcuts for faster labeling (see interface for shortcuts)

4. **Export Results**:
   - Click "Export" when finished
   - Choose your preferred export format

## Getting Help

- Official Documentation: https://labelstud.io/guide/
- If you encounter issues:
  1. Check the terminal for error messages
  2. Restart Label Studio
  3. Contact your supervisor if problems persist

## Shutting Down

- To stop Label Studio:
  1. Press `Ctrl+C` in the terminal where Label Studio is running
  2. Close the terminal window 