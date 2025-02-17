from app.services.assistant import KnowledgeAssistant
from app.config import settings
def main():
    try:
        knowledge_assistant = KnowledgeAssistant()
        knowledge_assistant.upload_knowledge_files()

        print("Asking about key insights...")
        knowledge_assistant.ask_question("What are the key insights from the interviews?")
        # Ask about key insights
        
        # knowledge_assistant.ask_question("What are the key insights?")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 