# Frontend Email Verification Implementation Guide

## 🎯 Overview

The authentication system now requires **email verification for ALL users** (both email/password and Google OAuth). Frontend needs to handle two main scenarios:

1. **After Signup**: Show "check your email" message instead of immediate login
2. **Email Link Click**: Handle verification when user clicks the email link

---

## 🔄 Changes Required

### **1. Signup Flow Changes**

#### **Before (Old Behavior)**:
- User signs up → Immediately gets tokens and logged in

#### **After (New Behavior)**:
- User signs up → Gets "check your email" message
- User must verify email before they can login

---

## 📋 Implementation Details

### **Step 1: Update Signup Responses**

Both signup endpoints now return different responses:

#### **Email Signup (`POST /auth/signup`)**
```json
// NEW Response Format
{
  "message": "Registration successful! Please check your email to verify your account.",
  "email": "user@example.com",
  "emailSent": true
}
```

#### **Google OAuth (`POST /auth/google`)**
```json
// For NEW unverified users
{
  "message": "Registration successful! Please check your email to verify your account before logging in.",
  "email": "user@example.com",
  "isNewUser": true,
  "provider": "google"
}

// For EXISTING verified users (login success)
{
  "user": { ... }, 
  "token": "...",
  "refreshToken": "...",
  "isNewUser": false
}
```

### **Step 2: Update Signup Components**

#### **React Example: Email Signup**
```jsx
const EmailSignup = () => {
  const [signupStatus, setSignupStatus] = useState('idle'); // idle, loading, success, error
  const [userEmail, setUserEmail] = useState('');

  const handleSignup = async (formData) => {
    setSignupStatus('loading');
    
    try {
      const response = await fetch('/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSignupStatus('success');
        setUserEmail(data.email);
        // DON'T redirect to dashboard - show verification message
      } else {
        setSignupStatus('error');
        setError(data.detail);
      }
    } catch (error) {
      setSignupStatus('error');
    }
  };

  // Show verification message instead of redirect
  if (signupStatus === 'success') {
    return <EmailVerificationPending email={userEmail} />;
  }

  return (
    // Your signup form JSX
  );
};
```

#### **React Example: Google OAuth**
```jsx
const GoogleOAuth = () => {
  const handleGoogleSignup = async (googleToken, userInfo) => {
    try {
      const response = await fetch('/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          googleToken,
          userInfo
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (data.user && data.token) {
          // Existing verified user - proceed with login
          localStorage.setItem('access_token', data.token);
          localStorage.setItem('refresh_token', data.refreshToken);
          router.push('/dashboard');
        } else {
          // New user needs email verification
          setShowVerificationMessage(true);
          setUserEmail(data.email);
        }
      }
    } catch (error) {
      // Handle error
    }
  };
};
```

### **Step 3: Create Verification Pending Component**

```jsx
const EmailVerificationPending = ({ email }) => {
  const [resending, setResending] = useState(false);
  const [resendStatus, setResendStatus] = useState('');

  const handleResendEmail = async () => {
    setResending(true);
    setResendStatus('');
    
    try {
      const response = await fetch('/auth/resend-verification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      
      if (response.ok) {
        setResendStatus('sent');
      } else {
        const data = await response.json();
        setResendStatus('error');
      }
    } catch (error) {
      setResendStatus('error');
    }
    
    setResending(false);
  };

  return (
    <div className="verification-pending">
      <div className="icon">📧</div>
      <h2>Check Your Email</h2>
      <p>
        We've sent a verification link to <strong>{email}</strong>
      </p>
      <p>Please click the link in your email to verify your account and complete registration.</p>
      
      <div className="resend-section">
        <p>Didn't receive the email?</p>
        <button 
          onClick={handleResendEmail}
          disabled={resending}
          className="resend-btn"
        >
          {resending ? 'Sending...' : 'Resend Email'}
        </button>
        
        {resendStatus === 'sent' && (
          <p className="success">Email sent! Please check your inbox.</p>
        )}
        {resendStatus === 'error' && (
          <p className="error">Failed to send email. Please try again.</p>
        )}
      </div>
      
      <div className="help-text">
        <p>Check your spam folder if you don't see it in your inbox.</p>
        <a href="/login">← Back to Login</a>
      </div>
    </div>
  );
};
```

### **Step 4: Create Email Verification Page**

Create a new route/page at `/auth/verify-email` to handle when users click the email link:

```jsx
// Route: /auth/verify-email
const EmailVerificationPage = () => {
  const [status, setStatus] = useState('loading'); // loading, success, error
  const [errorMessage, setErrorMessage] = useState('');
  const router = useRouter();
  
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    
    if (!token) {
      setStatus('error');
      setErrorMessage('No verification token provided.');
      return;
    }
    
    // Verify the token
    fetch('/auth/verify-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token })
    })
    .then(response => response.json())
    .then(data => {
      if (data.user && data.token) {
        // Success - store tokens and redirect
        localStorage.setItem('access_token', data.token);
        localStorage.setItem('refresh_token', data.refreshToken);
        setStatus('success');
        
        // Redirect after 2 seconds
        setTimeout(() => {
          router.push('/dashboard');
        }, 2000);
      } else {
        setStatus('error');
        setErrorMessage(data.detail || 'Verification failed.');
      }
    })
    .catch(error => {
      setStatus('error');
      setErrorMessage('An error occurred during verification.');
    });
  }, []);

  return (
    <div className="verification-page">
      {status === 'loading' && (
        <div className="loading">
          <div className="spinner"></div>
          <h2>Verifying your email...</h2>
          <p>Please wait while we verify your email address.</p>
        </div>
      )}
      
      {status === 'success' && (
        <div className="success">
          <div className="icon">✅</div>
          <h2>Email Verified Successfully!</h2>
          <p>Your email has been verified. Redirecting you to your dashboard...</p>
        </div>
      )}
      
      {status === 'error' && (
        <div className="error">
          <div className="icon">❌</div>
          <h2>Verification Failed</h2>
          <p>{errorMessage}</p>
          <div className="actions">
            <button onClick={() => router.push('/resend-verification')}>
              Request New Link
            </button>
            <button onClick={() => router.push('/login')}>
              Back to Login
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
```

### **Step 5: Update Login Error Handling**

Update login to handle verification errors:

```jsx
const Login = () => {
  const handleLogin = async (credentials) => {
    try {
      const response = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
      });
      
      const data = await response.json();
      
      if (response.ok) {
        // Success - store tokens and redirect
        localStorage.setItem('access_token', data.token);
        localStorage.setItem('refresh_token', data.refreshToken);
        router.push('/dashboard');
      } else if (response.status === 403) {
        // Email not verified
        setError('Please verify your email address before logging in.');
        setShowResendOption(true);
        setUserEmail(credentials.email);
      } else {
        setError(data.detail || 'Login failed');
      }
    } catch (error) {
      setError('Login failed');
    }
  };
};
```

---

## 🛣️ Routing Setup

Add these routes to your router:

```jsx
// React Router example
<Routes>
  <Route path="/auth/verify-email" component={EmailVerificationPage} />
  <Route path="/resend-verification" component={ResendVerificationPage} />
  {/* existing routes */}
</Routes>
```

---

## 🎨 Styling Suggestions

```css
.verification-pending {
  text-align: center;
  max-width: 500px;
  margin: 0 auto;
  padding: 2rem;
}

.verification-pending .icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.verification-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
}

.spinner {
  border: 3px solid #f3f3f3;
  border-top: 3px solid #007bff;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
```

---

## 📡 API Endpoints Reference

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/auth/signup` | POST | Email signup | `{email, password, name, ...}` | `{message, email, emailSent}` |
| `/auth/google` | POST | Google OAuth | `{googleToken, userInfo}` | `{message, email, ...}` or `{user, token, ...}` |
| `/auth/verify-email` | POST | Verify email | `{token}` | `{user, token, refreshToken}` |
| `/auth/resend-verification` | POST | Resend email | `{email}` | `{message, emailSent}` |
| `/auth/login` | POST | Login | `{email, password}` | `{user, token, refreshToken}` or `403 error` |

---

## ✅ Testing Checklist

- [ ] Email signup shows verification message (no immediate login)
- [ ] Google signup shows verification message for new users
- [ ] Google signup logs in existing verified users
- [ ] Email verification page works when clicking email links
- [ ] Resend verification email works
- [ ] Login blocks unverified users with helpful message
- [ ] All error states are handled gracefully
- [ ] Success states redirect properly
- [ ] Loading states show appropriate spinners

---

## 🚨 Important Notes

1. **No Immediate Login**: After signup, users MUST verify email first
2. **Token Storage**: Only store tokens after successful email verification
3. **Error Handling**: Always provide helpful messages and resend options
4. **Google Users**: Also require email verification (security enhancement)
5. **URL Handling**: The verification page must handle query parameters

---

## 💡 UX Recommendations

1. **Clear Messaging**: Always explain what the user needs to do next
2. **Resend Options**: Provide easy way to resend verification emails
3. **Loading States**: Show spinners during verification process
4. **Error Recovery**: Always provide a way to retry or get help
5. **Success Feedback**: Clear confirmation when verification succeeds

This implementation ensures maximum security while maintaining a smooth user experience! 