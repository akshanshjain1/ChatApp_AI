import os
from flask import Flask, request, jsonify
from pyngrok import ngrok
from langchain_community.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType

# Set environment variables for Cohere API key
os.environ['COHERE_API_KEY'] = "gUhIrRwRSOa1ScBubR8Q8UGxNCrgMvPRkM1XLQaP"

# Initialize Flask app
app = Flask(__name__)

# Initialize LangChain components
memory = ConversationBufferMemory()
llm = Cohere(temperature=0.7, max_tokens=500)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Create LangChain agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

# Expose public URL via ngrok
ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url.public_url}")

# Define API route
@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    data = request.json
    user_input = data.get('prompt')

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Generate response from LangChain agent
        agent_response = agent.invoke(user_input)
        return jsonify({"response": agent_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start Flask server
if __name__ == "__main__":
    app.run(port=5000)
