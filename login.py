from supabase import create_client, Client
from typing import Optional, Dict, Any
import os
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta, UTC
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
from google.auth.transport import requests
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
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        
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
            "email_verified": True if provider == "google" else False,
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
        googleId=user.get("google_id")
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

@router.post("/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    """User registration"""
    try:
        # Check if user already exists
        existing_user = get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create new user
        user = create_user(
            email=request.email,
            name=request.name,
            password=request.password,
            phone=request.phone,
            organization=request.organization
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
        logger.error(f"Error during signup: {str(e)}")
        raise HTTPException(status_code=500, detail="Signup failed")

@router.post("/google", response_model=AuthResponse)
async def google_auth(request: GoogleAuthRequest):
    """Google OAuth authentication"""
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
