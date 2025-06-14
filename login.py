from supabase import create_client, Client
from typing import Optional, Dict, Any
import os
import jwt
import hashlib
import secrets
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import ssl
import certifi
import requests
import json
from datetime import datetime, timedelta, UTC
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from utilities import logger

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(spb_url, spb_key)

# Initialize security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_DELTA = timedelta(hours=1)
REFRESH_TOKEN_SECRET = os.getenv("REFRESH_TOKEN_SECRET", "your-refresh-token-secret-change-in-production")
REFRESH_TOKEN_EXPIRES_DELTA = timedelta(days=7)

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# SendGrid Configuration
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Your App")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
# SSL Configuration - for environments with SSL issues
DISABLE_SSL_VERIFICATION = os.getenv("DISABLE_SSL_VERIFICATION", "false").lower() == "true"

# Email verification configuration
EMAIL_VERIFICATION_EXPIRES_DELTA = timedelta(hours=24)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Request Models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    organization: Optional[str] = None
    password: str
    confirmPassword: str
    
    @validator('confirmPassword')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class GoogleAuthRequest(BaseModel):
    googleToken: str
    userInfo: Dict[str, Any]

class RefreshTokenRequest(BaseModel):
    refreshToken: str

class LogoutRequest(BaseModel):
    refreshToken: Optional[str] = None

class ResendVerificationRequest(BaseModel):
    email: EmailStr

class VerifyEmailRequest(BaseModel):
    token: str

# Response Models
class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    orgId: Optional[str] = None
    role: str = "user"
    phone: Optional[str] = None
    organization: Optional[str] = None
    avatar: Optional[str] = None
    provider: str = "email"
    googleId: Optional[str] = None
    emailVerified: bool = False

class AuthResponse(BaseModel):
    user: UserResponse
    token: str
    refreshToken: str
    isNewUser: Optional[bool] = None

class TokenValidationResponse(BaseModel):
    valid: bool
    user: Optional[UserResponse] = None
    message: Optional[str] = None

# Utility Functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.now(UTC) + JWT_EXPIRES_DELTA
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    """Create JWT refresh token"""
    to_encode = {"user_id": user_id, "type": "refresh"}
    expire = datetime.now(UTC) + REFRESH_TOKEN_EXPIRES_DELTA
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, REFRESH_TOKEN_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str, secret: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = verify_token(token, JWT_SECRET)
    
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    # Get user from database
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

def verify_google_token(token: str) -> dict:
    """Verify Google OAuth token"""
    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
        
        return idinfo
    except ValueError as e:
        logger.error(f"Google token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid Google token")

def generate_user_id() -> str:
    """Generate a unique user ID"""
    return f"user_{secrets.token_urlsafe(16)}"

def generate_org_id() -> str:
    """Generate a unique organization ID"""
    return f"org_{secrets.token_urlsafe(16)}"

def generate_verification_token() -> str:
    """Generate a unique email verification token"""
    return secrets.token_urlsafe(32)

def send_verification_email(email: str, name: str, token: str) -> bool:
    """Send email verification email using SendGrid API"""
    try:
        if not SENDGRID_API_KEY:
            logger.error("SendGrid API key not configured")
            return False
        
        if not EMAIL_FROM:
            logger.error("EMAIL_FROM not configured")
            return False
        
        verification_url = f"{BASE_URL}/auth/verify-email?token={token}"
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px;">
                <h2 style="color: #333; text-align: center;">Welcome to our platform!</h2>
                <p style="font-size: 16px; color: #555;">Hi {name},</p>
                <p style="font-size: 16px; color: #555;">
                    Thank you for signing up! Please verify your email address by clicking the button below:
                </p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{verification_url}" 
                       style="background-color: #007bff; color: white; padding: 12px 30px; 
                              text-decoration: none; border-radius: 5px; font-weight: bold;">
                        Verify Email Address
                    </a>
                </div>
                <p style="font-size: 14px; color: #666;">
                    If the button doesn't work, you can copy and paste this URL into your browser:<br>
                    <a href="{verification_url}">{verification_url}</a>
                </p>
                <p style="font-size: 12px; color: #999; margin-top: 30px;">
                    This link will expire in 24 hours. If you didn't create an account, please ignore this email.
                </p>
            </div>
        </body>
        </html>
        """
        
        # Create plain text content
        text_content = f"""
        Welcome to our platform!
        
        Hi {name},
        
        Thank you for signing up! Please verify your email address by clicking the link below:
        {verification_url}
        
        This link will expire in 24 hours. If you didn't create an account, please ignore this email.
        """
        
        # Create SendGrid email object
        message = Mail(
            from_email=(EMAIL_FROM, EMAIL_FROM_NAME),
            to_emails=email,
            subject="Verify your email address",
            plain_text_content=text_content,
            html_content=html_content
        )
        
        # Send email via SendGrid API with robust SSL handling
        def send_with_sendgrid_api():
            """Send email using SendGrid REST API directly for better SSL control"""
            url = "https://api.sendgrid.com/v3/mail/send"
            
            # Convert SendGrid Mail object to API payload
            email_data = {
                "personalizations": [{
                    "to": [{"email": email}],
                    "subject": "Verify your email address"
                }],
                "from": {
                    "email": EMAIL_FROM,
                    "name": EMAIL_FROM_NAME
                },
                "content": [
                    {
                        "type": "text/plain",
                        "value": text_content
                    },
                    {
                        "type": "text/html", 
                        "value": html_content
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json"
            }
            
            return requests.post(url, json=email_data, headers=headers)
        
        try:
            if DISABLE_SSL_VERIFICATION:
                logger.warning("SSL verification disabled - using unverified HTTPS")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = send_with_sendgrid_api()
                response.requests_kwargs = {'verify': False}
            else:
                # Try with default SSL settings
                try:
                    sg = SendGridAPIClient(api_key=SENDGRID_API_KEY)
                    response = sg.send(message)
                except Exception as ssl_error:
                    if "SSL" in str(ssl_error) or "certificate" in str(ssl_error).lower():
                        logger.warning(f"SendGrid client SSL error: {ssl_error}")
                        logger.info("Trying direct API call with custom SSL handling...")
                        
                        try:
                            # Try with explicit certificate bundle
                            response = send_with_sendgrid_api()
                            response.verify = certifi.where()
                        except Exception as api_error:
                            if "SSL" in str(api_error) or "certificate" in str(api_error).lower():
                                logger.error(f"Direct API also failed: {api_error}")
                                logger.warning("Using unverified SSL as last resort")
                                
                                import urllib3
                                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                                response = requests.post(
                                    "https://api.sendgrid.com/v3/mail/send",
                                    json={
                                        "personalizations": [{
                                            "to": [{"email": email}],
                                            "subject": "Verify your email address"
                                        }],
                                        "from": {"email": EMAIL_FROM, "name": EMAIL_FROM_NAME},
                                        "content": [
                                            {"type": "text/plain", "value": text_content},
                                            {"type": "text/html", "value": html_content}
                                        ]
                                    },
                                    headers={
                                        "Authorization": f"Bearer {SENDGRID_API_KEY}",
                                        "Content-Type": "application/json"
                                    },
                                    verify=False
                                )
                            else:
                                raise api_error
                    else:
                        raise ssl_error
        except Exception as e:
            logger.error(f"All SendGrid methods failed: {str(e)}")
            raise e
        
        logger.info(f"SendGrid response: Status {response.status_code}")
        logger.info(f"Verification email sent to {email} via SendGrid")
        
        return response.status_code in [200, 201, 202]
        
    except Exception as e:
        logger.error(f"Error sending verification email via SendGrid: {str(e)}")
        return False

# Database Functions
def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email from database"""
    try:
        response = supabase.table("users").select("*").eq("email", email).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error getting user by email: {str(e)}")
        return None

def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID from database"""
    try:
        response = supabase.table("users").select("*").eq("id", user_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error getting user by ID: {str(e)}")
        return None

def get_user_by_google_id(google_id: str) -> Optional[dict]:
    """Get user by Google ID from database"""
    try:
        response = supabase.table("users").select("*").eq("google_id", google_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error getting user by Google ID: {str(e)}")
        return None

def create_user(email: str, name: str, password: str = None, phone: str = None, 
                organization: str = None, provider: str = "email", 
                google_id: str = None, avatar: str = None) -> dict:
    """Create a new user in database"""
    try:
        user_id = generate_user_id()
        org_id = generate_org_id() if organization else None
        
        user_data = {
            "id": user_id,
            "email": email,
            "name": name,
            "phone": phone,
            "organization": organization,
            "org_id": org_id,
            "role": "user",
            "avatar": avatar,
            "provider": provider,
            "google_id": google_id,
            "email_verified": False,  # All users require email verification
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat()
        }
        
        if password:
            user_data["password_hash"] = hash_password(password)
        
        response = supabase.table("users").insert(user_data).execute()
        
        if response.data:
            return response.data[0]
        
        raise Exception("Failed to create user")
    
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise

def store_refresh_token(user_id: str, token: str) -> bool:
    """Store refresh token in database"""
    try:
        token_id = f"rt_{secrets.token_urlsafe(16)}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        token_data = {
            "id": token_id,
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": (datetime.now(UTC) + REFRESH_TOKEN_EXPIRES_DELTA).isoformat(),
            "created_at": datetime.now(UTC).isoformat()
        }
        
        response = supabase.table("refresh_tokens").insert(token_data).execute()
        return bool(response.data)
    
    except Exception as e:
        logger.error(f"Error storing refresh token: {str(e)}")
        return False

def verify_refresh_token(token: str) -> Optional[dict]:
    """Verify refresh token and return user"""
    try:
        payload = verify_token(token, REFRESH_TOKEN_SECRET)
        
        if payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("user_id")
        if not user_id:
            return None
        
        # Check if token exists in database
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        response = supabase.table("refresh_tokens").select("*").eq("user_id", user_id).eq("token_hash", token_hash).execute()
        
        if not response.data:
            return None
        
        # Get user
        user = get_user_by_id(user_id)
        return user
    
    except Exception as e:
        logger.error(f"Error verifying refresh token: {str(e)}")
        return None

def invalidate_refresh_token(token: str) -> bool:
    """Invalidate refresh token"""
    try:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        response = supabase.table("refresh_tokens").delete().eq("token_hash", token_hash).execute()
        return True
    except Exception as e:
        logger.error(f"Error invalidating refresh token: {str(e)}")
        return False

def invalidate_user_refresh_tokens(user_id: str) -> bool:
    """Invalidate all refresh tokens for a user"""
    try:
        response = supabase.table("refresh_tokens").delete().eq("user_id", user_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error invalidating user refresh tokens: {str(e)}")
        return False

def create_verification_token(user_id: str, email: str) -> str:
    """Create and store email verification token"""
    try:
        token = generate_verification_token()
        token_id = f"evt_{secrets.token_urlsafe(16)}"
        
        token_data = {
            "id": token_id,
            "user_id": user_id,
            "token": token,
            "email": email,
            "expires_at": (datetime.now(UTC) + EMAIL_VERIFICATION_EXPIRES_DELTA).isoformat(),
            "created_at": datetime.now(UTC).isoformat()
        }
        
        response = supabase.table("email_verification_tokens").insert(token_data).execute()
        
        if response.data:
            return token
        
        raise Exception("Failed to create verification token")
    
    except Exception as e:
        logger.error(f"Error creating verification token: {str(e)}")
        raise

def verify_email_token(token: str) -> Optional[dict]:
    """Verify email verification token and return user"""
    try:
        logger.info(f"Attempting to verify token: {token}")
        
        # Get token from database
        response = supabase.table("email_verification_tokens").select("*").eq("token", token).execute()
        
        if not response.data:
            logger.error(f"Token not found in database: {token}")
            return None
        
        token_data = response.data[0]
        logger.info(f"Token data found: {token_data}")
        
        # Check if token is expired
        expires_at = datetime.fromisoformat(token_data["expires_at"].replace("Z", "+00:00"))
        current_time = datetime.now(UTC)
        logger.info(f"Token expires at: {expires_at}, Current time: {current_time}")
        
        if current_time > expires_at:
            logger.error(f"Token expired. Expires: {expires_at}, Now: {current_time}")
            return None
        
        # Check if already verified
        if token_data.get("verified_at"):
            logger.info(f"Token was already used at: {token_data.get('verified_at')}")
            # Check if the user is actually verified
            user_id = token_data["user_id"]
            user = get_user_by_id(user_id)
            if user and user.get("email_verified"):
                logger.info(f"User {user['email']} is already verified - returning user for seamless login")
                return user  # Return the already-verified user
            else:
                logger.error(f"Token was used but user is not verified - data inconsistency")
                return None
        
        # Get user
        user_id = token_data["user_id"]
        logger.info(f"Looking up user with ID: {user_id}")
        user = get_user_by_id(user_id)
        if not user:
            logger.error(f"User not found with ID: {user_id}")
            return None
        
        logger.info(f"User found: {user['email']} (ID: {user['id']})")
        
        # Mark user as verified
        logger.info("Updating user as verified...")
        user_update_response = supabase.table("users").update({
            "email_verified": True,
            "updated_at": datetime.now(UTC).isoformat()
        }).eq("id", user["id"]).execute()
        
        logger.info(f"User update response: {user_update_response}")
        
        # Mark token as used
        logger.info("Marking token as used...")
        token_update_response = supabase.table("email_verification_tokens").update({
            "verified_at": datetime.now(UTC).isoformat()
        }).eq("token", token).execute()
        
        logger.info(f"Token update response: {token_update_response}")
        
        # Update user object
        user["email_verified"] = True
        
        logger.info("Email verification successful!")
        return user
    
    except Exception as e:
        logger.error(f"Error verifying email token: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_pending_verification_token(email: str) -> Optional[dict]:
    """Get pending verification token for email"""
    try:
        response = supabase.table("email_verification_tokens").select("*").eq("email", email).is_("verified_at", "null").execute()
        
        if response.data:
            # Return the most recent token
            tokens = sorted(response.data, key=lambda x: x["created_at"], reverse=True)
            return tokens[0]
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting pending verification token: {str(e)}")
        return None

def user_to_response(user: dict) -> UserResponse:
    """Convert database user to response model"""
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        orgId=user.get("org_id"),
        role=user.get("role", "user"),
        phone=user.get("phone"),
        organization=user.get("organization"),
        avatar=user.get("avatar"),
        provider=user.get("provider", "email"),
        googleId=user.get("google_id"),
        emailVerified=user.get("email_verified", False)
    )

# Authentication Endpoints

@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Standard email/password authentication"""
    try:
        # Get user by email
        user = get_user_by_email(request.email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not user.get("password_hash"):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if not verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check email verification for all users
        if not user.get("email_verified"):
            raise HTTPException(
                status_code=403, 
                detail="Email not verified. Please check your email and click the verification link."
            )
        
        # Create tokens
        access_token = create_access_token({"user_id": user["id"], "email": user["email"]})
        refresh_token = create_refresh_token(user["id"])
        
        # Store refresh token
        store_refresh_token(user["id"], refresh_token)
        
        return AuthResponse(
            user=user_to_response(user),
            token=access_token,
            refreshToken=refresh_token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/signup")
async def signup(request: SignupRequest):
    """User registration with email verification"""
    try:
        # Check if user already exists
        existing_user = get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create new user (email_verified = False by default)
        user = create_user(
            email=request.email,
            name=request.name,
            password=request.password,
            phone=request.phone,
            organization=request.organization
        )
        
        # Create verification token
        verification_token = create_verification_token(user["id"], request.email)
        
        # Send verification email
        email_sent = send_verification_email(request.email, request.name, verification_token)
        
        if not email_sent:
            logger.error("Failed to send verification email")
            # Don't fail signup if email fails, just log it
        
        return {
            "message": "Registration successful! Please check your email to verify your account.",
            "email": request.email,
            "emailSent": email_sent
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        raise HTTPException(status_code=500, detail="Signup failed")

@router.post("/google")
async def google_auth(request: GoogleAuthRequest):
    """Google OAuth authentication with email verification"""
    try:
        # Verify Google token
        google_user_info = verify_google_token(request.googleToken)
        
        # Extract user info
        google_id = google_user_info["sub"]
        email = google_user_info["email"]
        name = google_user_info.get("name", "")
        avatar = google_user_info.get("picture", "")
        
        # Check if user already exists
        user = get_user_by_google_id(google_id)
        is_new_user = False
        
        if not user:
            # Check if user exists with same email
            user = get_user_by_email(email)
            if user:
                # Link Google account to existing user
                supabase.table("users").update({
                    "google_id": google_id,
                    "provider": "google",
                    "avatar": avatar,
                    "updated_at": datetime.now(UTC).isoformat()
                }).eq("id", user["id"]).execute()
                user["google_id"] = google_id
                user["provider"] = "google"
                user["avatar"] = avatar
                
                # If user is not verified, send verification email
                if not user.get("email_verified"):
                    verification_token = create_verification_token(user["id"], email)
                    send_verification_email(email, name, verification_token)
            else:
                # Create new user
                user = create_user(
                    email=email,
                    name=name,
                    provider="google",
                    google_id=google_id,
                    avatar=avatar
                )
                is_new_user = True
                
                # Send verification email for new Google user
                verification_token = create_verification_token(user["id"], email)
                email_sent = send_verification_email(email, name, verification_token)
        
        # Check if user is verified
        if not user.get("email_verified"):
            return {
                "message": "Registration successful! Please check your email to verify your account before logging in.",
                "email": email,
                "isNewUser": is_new_user,
                "provider": "google"
            }
        
        # User is verified, proceed with login
        # Create tokens
        access_token = create_access_token({"user_id": user["id"], "email": user["email"]})
        refresh_token = create_refresh_token(user["id"])
        
        # Store refresh token
        store_refresh_token(user["id"], refresh_token)
        
        return AuthResponse(
            user=user_to_response(user),
            token=access_token,
            refreshToken=refresh_token,
            isNewUser=is_new_user
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during Google authentication: {str(e)}")
        raise HTTPException(status_code=500, detail="Google authentication failed")

@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token"""
    try:
        # Verify refresh token
        user = verify_refresh_token(request.refreshToken)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Create new tokens
        access_token = create_access_token({"user_id": user["id"], "email": user["email"]})
        new_refresh_token = create_refresh_token(user["id"])
        
        # Invalidate old refresh token and store new one
        invalidate_refresh_token(request.refreshToken)
        store_refresh_token(user["id"], new_refresh_token)
        
        return AuthResponse(
            user=user_to_response(user),
            token=access_token,
            refreshToken=new_refresh_token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during token refresh: {str(e)}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

@router.post("/validate", response_model=TokenValidationResponse)
async def validate_token(current_user: dict = Depends(get_current_user)):
    """Validate access token"""
    try:
        return TokenValidationResponse(
            valid=True,
            user=user_to_response(current_user)
        )
    except HTTPException as e:
        return TokenValidationResponse(
            valid=False,
            message=e.detail
        )
    except Exception as e:
        logger.error(f"Error during token validation: {str(e)}")
        return TokenValidationResponse(
            valid=False,
            message="Token validation failed"
        )

@router.post("/logout")
async def logout(request: LogoutRequest, current_user: dict = Depends(get_current_user)):
    """Invalidate user session"""
    try:
        # Invalidate specific refresh token if provided
        if request.refreshToken:
            invalidate_refresh_token(request.refreshToken)
        else:
            # Invalidate all refresh tokens for user
            invalidate_user_refresh_tokens(current_user["id"])
        
        return {"message": "Logged out successfully"}
    
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

# Helper endpoint for getting current user info
@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return user_to_response(current_user)

@router.post("/verify-email", response_model=AuthResponse)
async def verify_email(request: VerifyEmailRequest):
    """Verify email address using token"""
    try:
        logger.info(f"Email verification endpoint called with token: {request.token}")
        
        # Verify the token
        user = verify_email_token(request.token)
        if not user:
            logger.error(f"Token verification failed for token: {request.token}")
            raise HTTPException(status_code=400, detail="Invalid or expired verification token")
        
        logger.info(f"Token verification successful for user: {user['email']}")
        
        # Create tokens for the newly verified user
        access_token = create_access_token({"user_id": user["id"], "email": user["email"]})
        refresh_token = create_refresh_token(user["id"])
        
        # Store refresh token
        store_refresh_token(user["id"], refresh_token)
        
        logger.info(f"Created new tokens for verified user: {user['email']}")
        
        return AuthResponse(
            user=user_to_response(user),
            token=access_token,
            refreshToken=refresh_token
        )
    
    except HTTPException as e:
        logger.error(f"HTTP Exception during email verification: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during email verification: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Email verification failed")

# Debug endpoint - remove in production
@router.get("/debug/token/{token}")
async def debug_token(token: str):
    """Debug endpoint to check token status"""
    try:
        response = supabase.table("email_verification_tokens").select("*").eq("token", token).execute()
        
        if not response.data:
            return {"error": "Token not found"}
        
        token_data = response.data[0]
        
        # Check expiration
        expires_at = datetime.fromisoformat(token_data["expires_at"].replace("Z", "+00:00"))
        current_time = datetime.now(UTC)
        is_expired = current_time > expires_at
        
        # Check user
        user = get_user_by_id(token_data["user_id"])
        
        return {
            "token_found": True,
            "token_data": token_data,
            "is_expired": is_expired,
            "expires_at": expires_at.isoformat(),
            "current_time": current_time.isoformat(),
            "user_exists": user is not None,
            "user_email": user.get("email") if user else None,
            "user_verified": user.get("email_verified") if user else None
        }
    except Exception as e:
        return {"error": str(e)}

@router.post("/resend-verification")
async def resend_verification(request: ResendVerificationRequest):
    """Resend email verification"""
    try:
        # Check if user exists
        user = get_user_by_email(request.email)
        if not user:
            # Don't reveal if email exists or not for security
            return {"message": "If the email exists, a verification link has been sent."}
        
        # Check if already verified
        if user.get("email_verified"):
            raise HTTPException(status_code=400, detail="Email is already verified")
        
        # All users now require email verification, so remove provider check
        
        # Check for existing pending token (to prevent spam)
        pending_token = get_pending_verification_token(request.email)
        if pending_token:
            # Check if token was created recently (less than 1 minute ago)
            created_at = datetime.fromisoformat(pending_token["created_at"].replace("Z", "+00:00"))
            if datetime.now(UTC) - created_at < timedelta(minutes=1):
                raise HTTPException(status_code=429, detail="Verification email was sent recently. Please wait before requesting another.")
        
        # Create new verification token
        verification_token = create_verification_token(user["id"], request.email)
        
        # Send verification email
        email_sent = send_verification_email(request.email, user["name"], verification_token)
        
        return {
            "message": "If the email exists, a verification link has been sent.",
            "emailSent": email_sent
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resending verification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to resend verification email")
