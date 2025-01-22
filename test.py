from StreamingTTS import StreamingTTS
from langchain.llms import Ollama

# Initialize the models
llm = Ollama(model="mistral")  # Replace with your desired Ollama model
streaming_tts = StreamingTTS()

def chat_loop():
    print("Chatbot started! Type 'stop' to exit.")

    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == "stop":
            print("Stopping chatbot...")
            streaming_tts.stop_stream()
            break

        # Generate response from Ollama
        response = llm.invoke(user_input)
        print("\nAI:", response)

        # Stream the response as speech
        try:
            streaming_tts.generate_and_stream(
                text=response,
                speaker_wav="myvoices/Sample1.wav",  # Optional for voice cloning
                language="en"
            )
        except KeyboardInterrupt:
            print("\nStopping TTS stream...")
            streaming_tts.stop_stream()

if __name__ == "__main__":
    chat_loop()
