import os
os.environ['COHERE_API_KEY'] = "gUhIrRwRSOa1ScBubR8Q8UGxNCrgMvPRkM1XLQaP"
from langchain_community.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.agents import initialize_agent, AgentType, load_tools
from flask import Flask, request, jsonify
from pyngrok import ngrok

# Initialize Flask app
app = Flask(__name__)

# Initialize LangChain components
memory = ConversationBufferMemory()
llm = Cohere(temperature=0.7, max_tokens=500)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

conversation = ConversationChain(llm=llm, memory=memory)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

# Expose public URL via ngrok
public_url = ngrok.connect(5000)
print("Public URL:", public_url.public_url)

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