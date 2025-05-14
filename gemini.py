from google import genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the client
client = genai.Client(api_key=API_KEY)

def chat_with_gemini():
    print("Welcome to Gemini Chat! Type 'exit' to end the conversation.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Ending chat session. Goodbye!")
            break
        
        try:
            # Generate content using the generate_content method
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=user_input
            )
            
            # Print the response
            print(f"\nGemini: {response.text}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Check if API key is available
    if not API_KEY:
        print("Error: Gemini API key not found!")
        print("Please create a .env file with your GEMINI_API_KEY=your_api_key")
        exit(1)
        
    # Start the chat
    chat_with_gemini()