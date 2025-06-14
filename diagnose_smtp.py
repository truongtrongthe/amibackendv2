#!/usr/bin/env python3
"""
SMTP Diagnostic Script for Production Servers
Run this on your production server to diagnose email sending issues
"""

import socket
import smtplib
import ssl
import os
from email.mime.text import MIMEText

def test_dns_resolution(hostname):
    """Test if we can resolve the SMTP server hostname"""
    try:
        ip = socket.gethostbyname(hostname)
        print(f"✅ DNS Resolution: {hostname} -> {ip}")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS Resolution Failed: {hostname} -> {e}")
        return False

def test_port_connectivity(hostname, port):
    """Test if we can connect to SMTP port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((hostname, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is reachable on {hostname}")
            return True
        else:
            print(f"❌ Port {port} is blocked on {hostname}")
            return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_smtp_auth(hostname, port, username, password, use_ssl=False, use_tls=True):
    """Test SMTP authentication"""
    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(hostname, port)
            print(f"✅ SSL connection established to {hostname}:{port}")
        else:
            server = smtplib.SMTP(hostname, port)
            print(f"✅ SMTP connection established to {hostname}:{port}")
            
            if use_tls:
                server.starttls()
                print(f"✅ TLS started successfully")
        
        server.login(username, password)
        print(f"✅ SMTP authentication successful")
        server.quit()
        return True
        
    except Exception as e:
        print(f"❌ SMTP test failed: {e}")
        return False

def test_email_send(hostname, port, username, password, from_email, to_email, use_ssl=False):
    """Test sending actual email"""
    try:
        msg = MIMEText("Test email from production server")
        msg['Subject'] = "SMTP Test Email"
        msg['From'] = from_email
        msg['To'] = to_email
        
        if use_ssl:
            server = smtplib.SMTP_SSL(hostname, port)
        else:
            server = smtplib.SMTP(hostname, port)
            server.starttls()
        
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Test email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False

def main():
    print("🔍 SMTP Diagnostic Tool for Production Servers")
    print("=" * 50)
    
    # Configuration
    servers_to_test = [
        {
            'name': 'Gmail TLS (Port 587)',
            'hostname': 'smtp.gmail.com',
            'port': 587,
            'use_ssl': False,
            'use_tls': True
        },
        {
            'name': 'Gmail SSL (Port 465)', 
            'hostname': 'smtp.gmail.com',
            'port': 465,
            'use_ssl': True,
            'use_tls': False
        },
        {
            'name': 'SendGrid',
            'hostname': 'smtp.sendgrid.net',
            'port': 587,
            'use_ssl': False,
            'use_tls': True
        }
    ]
    
    # Get credentials from environment
    email_username = os.getenv('EMAIL_USERNAME')
    email_password = os.getenv('EMAIL_PASSWORD')
    
    if not email_username or not email_password:
        print("⚠️  EMAIL_USERNAME and EMAIL_PASSWORD not found in environment")
        print("Set these environment variables and try again")
        return
    
    for config in servers_to_test:
        print(f"\n🧪 Testing {config['name']}")
        print("-" * 30)
        
        # Test DNS
        if not test_dns_resolution(config['hostname']):
            continue
            
        # Test port connectivity
        if not test_port_connectivity(config['hostname'], config['port']):
            continue
            
        # Test SMTP auth (skip SendGrid if no API key)
        if config['hostname'] == 'smtp.sendgrid.net' and not email_password.startswith('SG.'):
            print("⚠️  Skipping SendGrid (requires API key starting with 'SG.')")
            continue
            
        test_smtp_auth(
            config['hostname'], 
            config['port'], 
            email_username, 
            email_password,
            config['use_ssl'],
            config['use_tls']
        )
    
    print(f"\n📊 Diagnostic Summary")
    print("=" * 50)
    print("If all Gmail tests failed, your cloud provider likely blocks SMTP ports.")
    print("Recommendation: Use SendGrid, AWS SES, or another email service.")
    
    # Additional system info
    print(f"\n🖥️  System Information")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Python SSL Support: {ssl.OPENSSL_VERSION}")
    
    # Test environment variables
    print(f"\n🔧 Environment Variables")
    env_vars = ['EMAIL_SMTP_SERVER', 'EMAIL_SMTP_PORT', 'EMAIL_USERNAME', 'BASE_URL']
    for var in env_vars:
        value = os.getenv(var, 'NOT SET')
        print(f"{var}: {value}")

if __name__ == "__main__":
    main() 