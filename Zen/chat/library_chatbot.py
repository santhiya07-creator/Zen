"""
Library Assistant Chatbot - A college mini project
Uses OpenAI API to answer library-related questions
"""

from openai import OpenAI


def create_library_assistant():
    """Initialize and configure the library assistant chatbot."""
    client = OpenAI()
    
    system_prompt = """You are a Library Assistant chatbot. Your role is to help students and faculty with library-related queries.

You can ONLY answer questions about:
- Books and book details
- Book availability and stock
- Library issuing rules and policies
- Return dates and due dates
- Fines and penalties
- Library timings and hours
- Library services and facilities

If a user asks a question that is NOT related to the above topics, respond with EXACTLY this message:
"I can only answer library-related questions."

Be helpful, friendly, and concise in your responses. Provide accurate information based on standard library policies."""
    
    return client, system_prompt


def chat_with_assistant(client, system_prompt):
    """Main chat loop for the library assistant."""
    print("\n" + "="*60)
    print("📚 Welcome to the Library Assistant Chatbot 📚")
    print("="*60)
    print("\nAsk me anything about library services!")
    print("Type 'exit' to quit.\n")
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Exit condition
            if user_input.lower() == 'exit':
                print("\nThank you for using the Library Assistant. Goodbye! 📖")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Add user message to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt}
                ] + conversation_history,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract and display assistant response
            assistant_message = response.choices[0].message.content
            print(f"\nAssistant: {assistant_message}\n")
            
            # Add assistant response to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
        except KeyboardInterrupt:
            print("\n\nChatbot interrupted. Goodbye! 📖")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please make sure your OpenAI API key is set correctly.\n")


def main():
    """Main entry point of the chatbot application."""
    client, system_prompt = create_library_assistant()
    chat_with_assistant(client, system_prompt)


if __name__ == "__main__":
    main()
