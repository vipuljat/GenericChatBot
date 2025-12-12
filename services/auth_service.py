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
            
            # Safely check for access_token
            if result and "access_token" in result:
                return result
            else:
                error_desc = result.get("error_description", "Unknown error") if result else "No result"
                print(f"Error acquiring token: {error_desc}")
                print(f"Full result: {result}")
                return None
        except Exception as e:
            print(f"Exception in get_token_from_code: {str(e)}")
            import traceback
            traceback.print_exc()
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
                    # Parse JSON safely
                    try:
                        user_data = response.json()
                        print(f"Successfully fetched user info. Keys: {list(user_data.keys())}")
                        
                        # Return the raw dictionary without any processing
                        # Do NOT access any keys here with brackets
                        return dict(user_data)
                    except Exception as json_error:
                        print(f"Error parsing JSON: {str(json_error)}")
                        print(f"Response text: {response.text}")
                        return None
                else:
                    print(f"Error fetching user info: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
        except Exception as e:
            print(f"Exception in get_user_info: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_jwt_token(self, user_data: Dict) -> str:
        """Create JWT token from user data"""
        try:
            expiration = datetime.utcnow() + timedelta(minutes=config.JWT_EXPIRATION_MINUTES)

            # Safely extract all fields - handle both formats
            user_id = user_data.get("id") or user_data.get("sub", "")
            email = user_data.get("email") or user_data.get("mail") or user_data.get("userPrincipalName", "")
            name = user_data.get("name") or user_data.get("displayName", "")

            payload = {
                "sub": user_id,
                "email": email,
                "name": name,
                "exp": expiration
            }

            token = jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)
            return token
        except Exception as e:
            print(f"Error creating JWT token: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
            return payload
        except JWTError as e:
            print(f"JWT Error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error verifying JWT: {str(e)}")
            return None