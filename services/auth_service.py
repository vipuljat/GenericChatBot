import msal
import httpx
from typing import Optional, Dict
from datetime import datetime, timedelta
from jose import jwt, JWTError
import config

class MicrosoftAuthService:
    def __init__(self):
        self.client_id = config.MICROSOFT_CLIENT_ID
        self.client_secret = config.MICROSOFT_CLIENT_SECRET
        self.tenant_id = config.MICROSOFT_TENANT_ID
        self.redirect_uri = config.MICROSOFT_REDIRECT_URI
        self.authority = config.MICROSOFT_AUTHORITY
        self.scope = ["User.Read"]
        
        self.msal_app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=self.authority,
            client_credential=self.client_secret
        )
    
    def get_authorization_url(self) -> str:
        """Generate the Microsoft login URL"""
        auth_url = self.msal_app.get_authorization_request_url(
            scopes=self.scope,
            redirect_uri=self.redirect_uri
        )
        return auth_url
    
    async def get_token_from_code(self, code: str) -> Optional[Dict]:
        """Exchange authorization code for access token"""
        try:
            result = self.msal_app.acquire_token_by_authorization_code(
                code,
                scopes=self.scope,
                redirect_uri=self.redirect_uri
            )
            
            if "access_token" in result:
                return result
            else:
                print(f"Error acquiring token: {result.get('error_description')}")
                return None
        except Exception as e:
            print(f"Exception in get_token_from_code: {str(e)}")
            return None
    
    async def get_user_info(self, access_token: str) -> Optional[Dict]:
        """Get user information from Microsoft Graph API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://graph.microsoft.com/v1.0/me",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Error fetching user info: {response.status_code}")
                    return None
        except Exception as e:
            print(f"Exception in get_user_info: {str(e)}")
            return None
    
    def create_jwt_token(self, user_data: Dict) -> str:
        """Create JWT token for session management"""
        expiration = datetime.utcnow() + timedelta(minutes=config.JWT_EXPIRATION_MINUTES)
        
        payload = {
            "sub": user_data["id"],
            "email": user_data["mail"] or user_data.get("userPrincipalName"),
            "name": user_data["displayName"],
            "exp": expiration
        }
        
        token = jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
            return payload
        except JWTError as e:
            print(f"JWT Error: {str(e)}")
            return None
