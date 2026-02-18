# Library Assistant Chatbot

A Python-based chatbot for a college mini project that assists with library-related queries using OpenAI's GPT-4 API.

## Features

- ✅ Answers questions about books, availability, issuing rules, return dates, and fines
- ✅ Rejects non-library questions with a standard response
- ✅ Uses OpenAI API with `gpt-4o-mini` model
- ✅ Maintains conversation history for context-aware responses
- ✅ Terminal-based interactive interface
- ✅ Clean and simple code suitable for a college project

## Requirements

- Python 3.8+
- OpenAI Python library
- Valid OpenAI API key

## Installation

1. **Clone or download the project:**
   ```bash
   cd c:\chat
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key:**
   
   **Windows (PowerShell):**
   ```powershell
   $env:OPENAI_API_KEY = "your-api-key-here"
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```
   
   **Linux/Mac:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

Run the chatbot:
```bash
python library_chatbot.py
```

Example interaction:
```
📚 Welcome to the Library Assistant Chatbot 📚
============================================================

Ask me anything about library services!
Type 'exit' to quit.

You: What are your library timings?
Assistant: Our library is open from 8:00 AM to 6:00 PM on weekdays and 9:00 AM to 5:00 PM on weekends...

You: What is the capital of France?
Assistant: I can only answer library-related questions.

You: exit
Thank you for using the Library Assistant. Goodbye! 📖
```

## How It Works

1. **System Prompt:** Defines the chatbot's role as a Library Assistant and sets guidelines for acceptable topics
2. **Conversation History:** Maintains context across multiple messages for coherent conversations
3. **User Input Loop:** Accepts user queries until they type 'exit'
4. **OpenAI Integration:** Uses the `gpt-4o-mini` model for fast and cost-effective responses
5. **Error Handling:** Gracefully handles API errors and keyboard interrupts

## Code Structure

- `create_library_assistant()` - Initializes the OpenAI client and system prompt
- `chat_with_assistant()` - Main chat loop handling user interactions
- `main()` - Entry point of the application

## Notes

- The chatbot uses `gpt-4o-mini` for balanced performance and cost
- Conversation history is maintained during the session for better context
- Empty inputs are skipped
- Press `Ctrl+C` to exit at any time

## Customization

To customize the chatbot's behavior:
- Edit the `system_prompt` in `create_library_assistant()` to change the assistant's tone or knowledge base
- Modify `temperature` parameter in the API call (0.0 = deterministic, 1.0 = creative)
- Adjust `max_tokens` to control response length

## License

This is a college project. Feel free to use and modify as needed.
