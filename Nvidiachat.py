from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


API_KEY = 'api_key'


llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct",
    api_key=API_KEY
)


embedder = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    model_type="passage",
    api_key=API_KEY 
)

def run_chatbot():
    print("NVIDIA ChatBot is running. Type 'exit' to quit.\n")
while True:
    user_input = input("Ask your question: ").strip()

    if user_input in ['exit', 'Exit', 'EXIT', 'quit', 'Quit', 'QUIT']:
        print("👋 Exiting ChatBot.")
        break

    try:
        response = llm.invoke(user_input)
        print("🧠 Answer:", response.content)
    except Exception as e:
        print("❌ An error occurred:", str(e))

if __name__ == "__main__":
    run_chatbot()
