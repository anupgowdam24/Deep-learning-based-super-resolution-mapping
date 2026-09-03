import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure app can find src
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    print("Starting SRM Pipeline Web App...")
    print("Open http://localhost:8000/static/index.html in your browser")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
