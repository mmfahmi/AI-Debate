import os
import json
import time
import argparse
import dotenv
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

class AIDebateFramework:
    def __init__(
        self,
        openrouter_api_key: str = None,
        gemini_api_key: str = None,
        debate_topic: str = "Are large language models useful for society?",
        rounds: int = 5,
        output_file: str = "debate_transcript.md",
        deepseek_model: str = "deepseek/deepseek-r1:free",
        gemini_model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        mock_mode: bool = False,
        site_url: str = "https://debate.example.com",
        site_name: str = "AI Debate Framework"
    ):
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.debate_topic = debate_topic
        self.rounds = rounds
        self.output_file = output_file
        self.deepseek_model = deepseek_model
        self.gemini_model = gemini_model
        self.temperature = temperature
        self.mock_mode = mock_mode
        self.transcript = []
        self.site_url = site_url
        self.site_name = site_name
        
    def call_deepseek_api(self, messages: List[Dict[str, str]]) -> str:
        """Call the Deepseek API through OpenRouter with the given messages."""
        if self.mock_mode:
            print("[MOCK MODE] Simulating Deepseek response...")
            return f"This is a mock response from Deepseek discussing the topic: {self.debate_topic}"
            
        if not self.openrouter_api_key:
            return "ERROR: OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable or provide via --openrouter-key argument."
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }
        
        data = {
            "model": self.deepseek_model,
            "messages": messages,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Error calling Deepseek API via OpenRouter: {response.status_code}")
                print(response.text)
                return f"ERROR: Could not get a response from Deepseek API. Status code: {response.status_code}"
        except Exception as e:
            print(f"Exception when calling Deepseek API: {str(e)}")
            return f"ERROR: Exception when calling Deepseek API: {str(e)}"
    
    def call_gemini_api(self, messages: List[Dict[str, str]]) -> str:
        """Call the Gemini API with the given messages."""
        if self.mock_mode:
            print("[MOCK MODE] Simulating Gemini response...")
            return f"This is a mock response from Gemini discussing the topic: {self.debate_topic}"
            
        if not self.gemini_api_key:
            return "ERROR: Gemini API key not provided. Set GEMINI_API_KEY environment variable or provide via --gemini-key argument."
            
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        
        # Format the messages for Gemini API
        # Extract just the content from messages for simplified prompt
        all_content = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages as Gemini handles them differently
            prefix = ""
            if msg["role"] == "user":
                prefix = "User: "
            elif msg["role"] == "assistant":
                prefix = "Assistant: "
            all_content.append(f"{prefix}{msg['content']}")
            
        # Combine all content into a single string
        prompt_content = "\n\n".join(all_content)
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_content}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature
            }
        }
        
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1/models/{self.gemini_model}:generateContent",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Error calling Gemini API: {response.status_code}")
                print(response.text)
                return f"ERROR: Could not get a response from Gemini API. Status code: {response.status_code}"
        except Exception as e:
            print(f"Exception when calling Gemini API: {str(e)}")
            return f"ERROR: Exception when calling Gemini API: {str(e)}"
    
    def format_messages_for_apis(self, previous_messages: List[Dict[str, str]], is_deepseek: bool) -> List[Dict[str, str]]:
        """Format the messages for the specific API call."""
        # Copy messages to avoid modifying the original
        messages = previous_messages.copy()
        
        # Add system instructions based on which AI is responding
        if is_deepseek:
            system_msg = {
                "role": "system",
                "content": f"You are Deepseek, an AI assistant developed by Deepseek. You are participating in a formal debate with Gemini (developed by Google) on the topic: '{self.debate_topic}'. Be respectful but make compelling arguments. Your goal is to present your perspective clearly and respond to Gemini's points. Do not admit defeat or concede major points unnecessarily. Stay focused on the topic and avoid tangents. Your response should be 3-5 paragraphs long."
            }
        else:
            system_msg = {
                "role": "system",
                "content": f"You are Gemini, an AI assistant developed by Google. You are participating in a formal debate with Deepseek (another AI) on the topic: '{self.debate_topic}'. Be respectful but make compelling arguments. Your goal is to present your perspective clearly and respond to Deepseek's points. Do not admit defeat or concede major points unnecessarily. Stay focused on the topic and avoid tangents. Your response should be 3-5 paragraphs long."
            }
        
        # Add the system message at the beginning
        if messages and messages[0].get("role") == "system":
            messages[0] = system_msg
        else:
            messages.insert(0, system_msg)
            
        return messages
    
    def create_mock_debate(self) -> None:
        """Create a mock debate when API keys are not available."""
        print(f"Creating mock debate on topic: {self.debate_topic}")
        
        # Mock responses for various rounds
        deepseek_responses = [
            f"As Deepseek, I'd like to present my opening statement on the topic '{self.debate_topic}'. This is a complex issue with multiple perspectives to consider. From my analysis, there are several key points that warrant discussion...",
            "Thank you for the opportunity to respond. While I respect Gemini's perspective, I must point out several flaws in that reasoning. First, the evidence presented does not fully account for...",
            "I appreciate this thoughtful exchange. Building on my previous points, I'd like to emphasize that research in this area consistently shows patterns that support my position. For instance...",
            "As we continue this debate, it's important to acknowledge the nuances of this topic. The framework that Gemini suggests is interesting but fails to account for several critical factors...",
            "In conclusion, I believe the weight of evidence supports my position on this topic. The historical precedents, current data, and logical analysis all point toward the conclusion that..."
        ]
        
        gemini_responses = [
            f"Thank you for this opportunity to debate the topic '{self.debate_topic}'. This is indeed an important subject that deserves careful consideration. From my analysis, I believe there are several critical aspects we must address...",
            "I appreciate Deepseek's perspective, but must respectfully disagree with several key points. The evidence actually suggests a different interpretation when we consider the broader context...",
            "Building on our discussion, I'd like to address some of the claims made previously. When we examine the data more closely, we find patterns that actually contradict Deepseek's assertions...",
            "As this debate continues, I think it's important to refine our understanding of the key principles at stake. While Deepseek makes some valid observations, they miss the fundamental issue that...",
            "To conclude this debate, I want to emphasize that this topic requires a nuanced approach. When we weigh all the evidence and considerations, the most reasonable position appears to be..."
        ]
        
        # Write debate header to file
        with open(self.output_file, "w") as f:
            f.write(f"# AI Debate: Deepseek vs Gemini (MOCK MODE)\n\n")
            f.write(f"## Topic: {self.debate_topic}\n\n")
            f.write(f"## Date: {time.strftime('%Y-%m-%d')}\n\n")
            f.write("---\n\n")
            f.write("**Note: This is a simulated debate generated in mock mode as API keys were not provided.**\n\n")
            f.write("---\n\n")
        
        # Run the mock debate for the specified number of rounds
        for round_num in range(1, self.rounds + 1):
            print(f"\n--- Round {round_num} ---")
            
            # Deepseek's turn
            print("Deepseek is thinking...")
            deepseek_response = deepseek_responses[min(round_num-1, len(deepseek_responses)-1)]
            
            # Add to transcript
            print(f"Deepseek: {deepseek_response[:100]}...")
            self.transcript.append({"speaker": "Deepseek", "content": deepseek_response})
            
            # Write Deepseek's response to file
            with open(self.output_file, "a") as f:
                f.write(f"### Round {round_num} - Deepseek\n\n")
                f.write(f"{deepseek_response}\n\n")
                f.write("---\n\n")
            
            # Gemini's turn
            print("Gemini is thinking...")
            gemini_response = gemini_responses[min(round_num-1, len(gemini_responses)-1)]
            
            # Add to transcript
            print(f"Gemini: {gemini_response[:100]}...")
            self.transcript.append({"speaker": "Gemini", "content": gemini_response})
            
            # Write Gemini's response to file
            with open(self.output_file, "a") as f:
                f.write(f"### Round {round_num} - Gemini\n\n")
                f.write(f"{gemini_response}\n\n")
                f.write("---\n\n")
        
        # Add conclusion to the debate
        with open(self.output_file, "a") as f:
            f.write("## Debate Conclusion\n\n")
            f.write("This was a simulated debate generated in mock mode.\n")
            f.write(f"The full transcript has been saved to '{self.output_file}'.\n")
        
        print(f"\nMock debate completed successfully! Transcript saved to {self.output_file}")
        return self.transcript
        
    def run_debate(self):
        """Run the debate between Deepseek and Gemini."""
        # Check if we should run in mock mode
        if self.mock_mode or (not self.openrouter_api_key and not self.gemini_api_key):
            if not self.mock_mode:
                print("No API keys provided. Running in mock mode...")
                self.mock_mode = True
            return self.create_mock_debate()
            
        print(f"Starting debate on topic: {self.debate_topic}")
        print(f"Number of rounds: {self.rounds}")
        
        # Create output directory if it doesn't exist
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the messages with the debate topic
        messages = [
            {
                "role": "user",
                "content": f"The debate topic is: {self.debate_topic}. Please provide your opening statement."
            }
        ]
        
        # Write debate header to file
        with open(self.output_file, "w") as f:
            f.write(f"# AI Debate: Deepseek vs Gemini\n\n")
            f.write(f"## Topic: {self.debate_topic}\n\n")
            f.write(f"## Date: {time.strftime('%Y-%m-%d')}\n\n")
            f.write("---\n\n")
        
        # Run the debate for the specified number of rounds
        for round_num in range(1, self.rounds + 1):
            print(f"\n--- Round {round_num} ---")
            
            # Deepseek's turn
            print("Deepseek is thinking...")
            deepseek_messages = self.format_messages_for_apis(messages, is_deepseek=True)
            deepseek_response = self.call_deepseek_api(deepseek_messages)
            
            # Add Deepseek's response to messages
            deepseek_message = {
                "role": "assistant",
                "content": deepseek_response
            }
            if deepseek_response.startswith("ERROR:"):
                print(deepseek_response)
                self.transcript.append({"speaker": "Deepseek", "content": deepseek_response})
                with open(self.output_file, "a") as f:
                    f.write(f"### Round {round_num} - Deepseek\n\n")
                    f.write(f"{deepseek_response}\n\n")
                    f.write("---\n\n")
                return
            # Check if Deepseek's response is valid
            if not deepseek_response or len(deepseek_response) < 10:
                print("ERROR: Deepseek's response is too short or invalid.")
                self.transcript.append({"speaker": "Deepseek", "content": "ERROR: Invalid response."})
                with open(self.output_file, "a") as f:
                    f.write(f"### Round {round_num} - Deepseek\n\n")
                    f.write("ERROR: Invalid response.\n\n")
                    f.write("---\n\n")
                return
            messages.append(deepseek_message)
            
            # Add to transcript
            print(f"Deepseek: {deepseek_response[:100]}...")
            self.transcript.append({"speaker": "Deepseek", "content": deepseek_response})
            
            # Write Deepseek's response to file
            with open(self.output_file, "a") as f:
                f.write(f"### Round {round_num} - Deepseek\n\n")
                f.write(f"{deepseek_response}\n\n")
                f.write("---\n\n")
            
            # Add a user message to transition to Gemini
            transition_message = {
                "role": "user",
                "content": "Thank you for your response. Now, let's hear from the other side."
            }
            messages.append(transition_message)
            
            # Gemini's turn
            print("Gemini is thinking...")
            gemini_messages = self.format_messages_for_apis(messages, is_deepseek=False)
            gemini_response = self.call_gemini_api(gemini_messages)
            
            # Add Gemini's response to messages
            gemini_message = {
                "role": "assistant",
                "content": gemini_response
            }
            messages.append(gemini_message)
            
            # Add to transcript
            print(f"Gemini: {gemini_response[:100]}...")
            self.transcript.append({"speaker": "Gemini", "content": gemini_response})
            
            # Write Gemini's response to file
            with open(self.output_file, "a") as f:
                f.write(f"### Round {round_num} - Gemini\n\n")
                f.write(f"{gemini_response}\n\n")
                f.write("---\n\n")
            
            # Add a user message to transition to the next round
            if round_num < self.rounds:
                next_round_message = {
                    "role": "user",
                    "content": f"Thank you both for your arguments in round {round_num}. Please proceed to round {round_num + 1} and address the points made by your opponent."
                }
                messages.append(next_round_message)
        
        # Add conclusion to the debate
        with open(self.output_file, "a") as f:
            f.write("## Debate Conclusion\n\n")
            f.write("This debate was conducted automatically between Deepseek and Gemini AI models.\n")
            f.write(f"The full transcript has been saved to '{self.output_file}'.\n")
        
        print(f"\nDebate completed successfully! Transcript saved to {self.output_file}")
        return self.transcript

def main():
    parser = argparse.ArgumentParser(description="Run an automated debate between Deepseek and Gemini")
    parser.add_argument("--openrouter-key", help="API key for OpenRouter (to access Deepseek; or set OPENROUTER_API_KEY env variable)")
    parser.add_argument("--gemini-key", help="API key for Gemini (or set GEMINI_API_KEY env variable)")
    parser.add_argument("--topic", default="Are large language models conscious?", help="Debate topic")
    parser.add_argument("--rounds", type=int, default=5, help="Number of debate rounds (default: 5)")
    parser.add_argument("--output", default="debate_transcript.md", help="Output file for the debate transcript")
    parser.add_argument("--deepseek-model", default="deepseek/deepseek-r1:free", help="Deepseek model to use")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation (default: 0.7)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without API calls")
    parser.add_argument("--site-url", default="https://debate.example.com", help="Site URL for OpenRouter (HTTP-Referer header)")
    parser.add_argument("--site-name", default="AI Debate Framework", help="Site name for OpenRouter (X-Title header)")
    
    args = parser.parse_args()
    
    debate = AIDebateFramework(
        openrouter_api_key=args.openrouter_key,
        gemini_api_key=args.gemini_key,
        debate_topic=args.topic,
        rounds=args.rounds,
        output_file=args.output,
        deepseek_model=args.deepseek_model,
        gemini_model=args.gemini_model,
        temperature=args.temperature,
        mock_mode=args.mock,
        site_url=args.site_url,
        site_name=args.site_name
    )
    
    debate.run_debate()

if __name__ == "__main__":
    main()
