import os
from getpass import getpass
from openai import OpenAI
from time import sleep

"""
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
"""

def setup_openai_api():
    """Setup OpenAI API with key from user input or environment"""
    # Always prompt for key if there's an error
    try:
        print("Please enter your OpenAI API key (it will not be displayed):")
        api_key = getpass()
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Create client
        client = OpenAI(api_key=api_key)
        print("OpenAI API key loaded successfully")
        
        # Test the API connection
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=5,
            store=True
        )
        print("API connection test successful!")
        return client
        
    except Exception as e:
        print(f"Error testing API connection: {str(e)}")
        # Clear the API key if there was an error
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        raise

def get_llm_response(client, prompt, max_retries=3, model="gpt-4o"):
    """Get response from OpenAI API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at text adventure games."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                store=True
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit, waiting {sleep_time} seconds...")
                sleep(sleep_time)
                continue
            
            print(f"Error calling OpenAI API: {str(e)}")
            if attempt < max_retries - 1:
                continue
            raise
            
    return None

def validate_response_format(response):
    """Check if response matches our expected format"""
    if not response:
        return False
        
    # Check for numbered actions
    if not any(line.strip().startswith(str(i)) for i in range(1, 10)):
        return False
        
    # Check for "Therefore, I choose:"
    if "Therefore, I choose:" not in response:
        return False
        
    return True
