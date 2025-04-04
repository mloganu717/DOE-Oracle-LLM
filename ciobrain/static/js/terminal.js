document.addEventListener("DOMContentLoaded", () => {
    const scrollContainer = document.getElementById("scroll-container");
    const input = document.getElementById("prompt");
    const ragToggle = document.getElementById("rag-toggle");

    // Function to format the message content
    const formatMessage = (content) => {
        // Split content into lines
        const lines = content.split('\n');
        return lines.map(line => {
            line = line.trim();
            if (line.match(/^\d+\./)) {
                // If line starts with a number and period, wrap in a div
                return `<div class="numbered-item">${line}</div>`;
            }
            return line;
        }).join('\n');
    };

    // Function to add a message to the terminal
    const addMessageToTerminal = (role, content) => {
        const message = document.createElement("div");
        message.className = role; // 'user' or 'assistant'
        
        if (role === 'user') {
            message.textContent = content;
        } else {
            // For assistant messages, preserve formatting
            message.innerHTML = formatMessage(content);
        }
        
        scrollContainer.appendChild(message);
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
        return message;
    };

    // Handle user input on Enter key
    input.addEventListener("keypress", async (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            const prompt = input.value.trim();
            if (!prompt) return;

            addMessageToTerminal("user", prompt);
            input.value = "";

            const loadingMessage = addMessageToTerminal("assistant", "Assistant is typing...");

            try {
                const response = await fetch("/customer/prompt", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt, use_rag: ragToggle.checked }),
                });

                if (!response.ok) throw new Error("Server error");

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                
                const assistantMessage = addMessageToTerminal("assistant", "");
                loadingMessage.remove();

                let accumulatedText = "";
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    accumulatedText += chunk;
                    assistantMessage.innerHTML = formatMessage(accumulatedText);
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                }
            } catch (error) {
                console.error("Error while streaming response:", error);
                loadingMessage.remove();
                addMessageToTerminal("error", "Error connecting to the server. Please try again.");
            }
        }
    });
});

