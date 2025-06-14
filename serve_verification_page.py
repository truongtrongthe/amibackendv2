# Add this to your main FastAPI app for local testing

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Include your auth router
app.include_router(router)  # Your existing auth router

# Serve static files (optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the email verification page
@app.get("/auth/verify-email")
async def serve_verification_page():
    """Serve the email verification HTML page"""
    return FileResponse("verify_email.html")

# For local testing - you can add a simple test endpoint
@app.get("/test-email/{email}")
async def test_send_verification_email(email: str):
    """Test endpoint to send verification email manually"""
    try:
        # Create a test user (for testing only)
        verification_token = generate_verification_token()
        email_sent = send_verification_email(email, "Test User", verification_token)
        
        return {
            "message": "Test email sent",
            "email": email,
            "emailSent": email_sent,
            "testToken": verification_token,  # Only for testing!
            "testUrl": f"http://localhost:8000/auth/verify-email?token={verification_token}"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 