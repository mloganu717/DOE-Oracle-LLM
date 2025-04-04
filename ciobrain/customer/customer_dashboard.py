import logging

class CustomerDashboard:
    """
    Facade class for customer dashboard
    """
    def __init__(self, mediator):
        self.chat_handler = ChatHandler(mediator=mediator)

    def process_prompt(self, prompt, use_rag=False):
        """
        Delegate chat prompt processing to the ChatHandler
        """
        return self.chat_handler.generate_response_stream(prompt, use_rag=use_rag)

class ChatHandler:
    def __init__(self, mediator):
        self.mediator = mediator
        self.chat_history = []

    def generate_response_stream(self, prompt, use_rag=False):
        """
        Generates a streaming response for a given prompt.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Append user prompt to chat history
        self.chat_history.append({"role": "user", "content": prompt})

        def generate():
            response_buffer = []

            # Prepare the conversation history to be passed to the mediator
            full_history = self.chat_history
            
            # Logging the conversation history for debugging
            logging.info(f"Full history being sent to mediator: {full_history}")

            # Stream the response using the mediator's stream function
            generator = self.mediator.stream(full_history, use_rag=use_rag)
            current_line = ""

            for chunk in generator:
                # Log each chunk received from the mediator
                logging.info(f"Chunk received from mediator: {chunk}")

                # Add chunk to buffer
                response_buffer.append(str(chunk))
                current_line += str(chunk)

                # If we have a complete sentence or numbered item, yield it
                if current_line.strip():
                    if current_line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        # For numbered items, add extra newlines
                        yield f"\n{current_line.strip()}\n"
                        current_line = ""
                    elif current_line.strip().endswith('.'):
                        # For complete sentences, add a newline
                        yield f"{current_line.strip()}\n"
                        current_line = ""

            # Yield any remaining text
            if current_line.strip():
                yield current_line.strip()

            # Once streaming is complete, add the final assistant response to the chat history
            final_response = "".join(response_buffer)
            self.chat_history.append({"role": "assistant", "content": final_response})
            logging.info(f"Final response added to history: {final_response}")

        return generate()
            
