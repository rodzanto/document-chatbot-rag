import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['Awsome23']).generate()
print(hashed_passwords)
