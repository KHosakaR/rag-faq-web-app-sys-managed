/**
 * Chat functionality for the RAG application.
 * 
 * This JavaScript handles the client-side functionality of the RAG application:
 * - Manages the chat UI (sending messages, displaying responses)
 * - Communicates with the FastAPI backend via fetch API
 * - Handles citations and displays them in a modal
 * - Manages error states and loading indicators
 * 
 * The chat interface supports:
 * 1. Free-form text input
 * 2. Quick-select question buttons
 * 3. Interactive citations from source documents
 * 4. Responsive design for various screen sizes
 */
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatHistory = document.getElementById('chat-history');
    const chatContainer = document.getElementById('chat-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    
    // Templates
    const emptyChatTemplate = document.getElementById('empty-chat-template');
    const userMessageTemplate = document.getElementById('user-message-template');
    const assistantMessageTemplate = document.getElementById('assistant-message-template');
    
    // Quick response buttons
    const btnPersonalInfo = document.getElementById('btn-personal-info');
    const btnWarranty = document.getElementById('btn-warranty');
    const btnCompany = document.getElementById('btn-company');
    
    // Chat history array
    let messages = [];
    
    // Initialize empty chat
    if (emptyChatTemplate) {
        const emptyContent = emptyChatTemplate.content.cloneNode(true);
        chatHistory.appendChild(emptyContent);
    }
    
    // Event listeners
    chatForm.addEventListener('submit', handleChatSubmit);
    chatInput.addEventListener('keydown', handleKeyDown);
    btnPersonalInfo.addEventListener('click', () => sendQuickQuestion("What does Contoso do with my personal information?"));
    btnWarranty.addEventListener('click', () => sendQuickQuestion("How do I file a warranty claim?"));
    btnCompany.addEventListener('click', () => sendQuickQuestion("Tell me about your company."));
    
    /**
     * Handles form submission when the user sends a message
     */
    function handleChatSubmit(e) {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (query && !isLoading()) {
            sendMessage(query);
        }
    }
    
    /**
     * Handles sending a message when Enter key is pressed
     */
    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const query = chatInput.value.trim();
            if (query && !isLoading()) {
                sendMessage(query);
            }
        }
    }
    
    /**
     * Sends a predefined quick question when a suggestion button is clicked
     */
    function sendQuickQuestion(text) {
        if (!isLoading()) {
            chatInput.value = text;
            sendMessage(text);
        }
    }
    
    /**
     * Checks if a request is currently loading
     */
    function isLoading() {
        return !loadingIndicator.classList.contains('d-none');
    }
    
    /**
     * Displays a user message in the chat interface
     */
    function addUserMessage(text) {
        // Clear empty chat template if this is the first message
        if (chatHistory.querySelector('.text-center')) {
            chatHistory.innerHTML = '';
        }
        
        const messageNode = userMessageTemplate.content.cloneNode(true);
        const messageContent = messageNode.querySelector('.message-content');
        messageContent.innerHTML = text.replace(/\n/g, '<br>');
        chatHistory.appendChild(messageNode);
        scrollToBottom();
    }
    
    /**
     * Displays an assistant message with citations in the chat interface
     * 
     * This function:
     * 1. Creates the HTML for the assistant's message
     * 2. Processes any citations returned from Azure AI Search
     * 3. Converts citation references [doc1], [doc2], etc. into clickable badges
     * 4. Sets up event handlers for citation badge clicks
     * 5. Adds the message to the chat history
     */
    function addAssistantMessage(content, citations) {
        const messageNode = assistantMessageTemplate.content.cloneNode(true);
        const messageContent = messageNode.querySelector('.message-content');
        const messageDiv = messageNode.querySelector('.card');
        
        // Create a unique ID for this message
        const messageId = 'msg-' + Date.now();
        messageDiv.setAttribute('id', messageId);
        
        // Create a message-specific citation data store
        const messageCitations = {};
        
        if (content && content.length > 0) {
            // Format content with citations if available
            let formattedContent = content;
            
            if (citations && citations.length > 0) {
                // Replace [doc1], [doc2], etc. with interactive citation links
                const pattern = /\[doc(\d+)\]/g;
                formattedContent = formattedContent.replace(pattern, (match, index) => {
                    const idx = parseInt(index);
                    if (idx > 0 && idx <= citations.length) {
                        const citation = citations[idx - 1];
                        const citationData = JSON.stringify({
                            title: citation.title || '',
                            content: citation.content || '',
                            filePath: citation.filePath || '',
                            url: citation.url || ''
                        });
                        
                        // Store citation data in this message's citations
                        messageCitations[idx] = citationData;
                        
                        // Create badge-style citation link
                        return `<a class="badge bg-primary rounded-pill" style="cursor: pointer;" data-message-id="${messageId}" data-index="${idx}">${idx}</a>`;
                    }
                    return match;
                });
            }
            
            messageContent.innerHTML = formattedContent.replace(/\n/g, '<br>');
            
            // Store the message citations as a data attribute
            messageDiv.setAttribute('data-citations', JSON.stringify(messageCitations));
            
            // Add click listeners for citation badges
            setTimeout(() => {
                const badges = messageContent.querySelectorAll('.badge[data-index]');
                
                badges.forEach(badge => {
                    badge.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        const messageId = this.getAttribute('data-message-id');
                        const idx = this.getAttribute('data-index');
                        
                        // Get the message element
                        const messageElement = document.getElementById(messageId);
                        
                        // Get this message's citations
                        const messageCitations = JSON.parse(messageElement.getAttribute('data-citations') || '{}');
                        const citationData = JSON.parse(messageCitations[idx]);
                        
                        // Show citation modal
                        showCitationModal(citationData);
                    });
                });
            }, 100);
        }
        
        chatHistory.appendChild(messageNode);
        scrollToBottom();
    }
    
    /**
     * Shows a modal with citation details
     */
    function showCitationModal(citationData) {
        // Remove any existing modal
        const existingOverlay = document.querySelector('.citation-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }
        
        // Format the citation content for better display
        let formattedContent = citationData.content || 'No content available';
        
        // Create overlay and modal
        const overlay = document.createElement('div');
        overlay.className = 'citation-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        overlay.setAttribute('aria-labelledby', 'citation-modal-title');
        
        overlay.innerHTML = `
            <div class="citation-modal">
                <div class="citation-modal-header">
                    <h5 class="citation-modal-title" id="citation-modal-title">${citationData.title || 'Citation'}</h5>
                    <button type="button" class="citation-close-button" aria-label="Close">&times;</button>
                </div>
                <div class="citation-modal-body">
                    <pre class="citation-content">${formattedContent}</pre>
                    ${citationData.filePath ? `<div class="citation-source mt-3"><strong>Source:</strong> ${citationData.filePath}</div>` : ''}
                    ${citationData.url ? `<div class="citation-url mt-2"><strong>URL:</strong> <a href="${citationData.url}" target="_blank" rel="noopener noreferrer">${citationData.url}</a></div>` : ''}
                </div>
            </div>
        `;
        
        // Add overlay to the document
        document.body.appendChild(overlay);
        
        // Set focus on the modal container for keyboard navigation
        const modal = overlay.querySelector('.citation-modal');
        modal.focus();
        
        // Handle close button click
        const closeButton = overlay.querySelector('.citation-close-button');
        closeButton.addEventListener('click', () => {
            overlay.remove();
        });
        
        // Close modal when clicking outside
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
        
        // Close modal on escape key
        document.addEventListener('keydown', function closeOnEscape(e) {
            if (e.key === 'Escape') {
                overlay.remove();
                document.removeEventListener('keydown', closeOnEscape);
            }
        });
    }
    
    /**
     * Displays an error message
     */
    function showError(text) {
        errorMessage.textContent = text;
        errorContainer.classList.remove('d-none');
    }
    
    /**
     * Hides the error message
     */
    function hideError() {
        errorContainer.classList.add('d-none');
    }
    
    /**
     * Shows the loading indicator
     */
    function showLoading() {
        loadingIndicator.classList.remove('d-none');
        sendButton.disabled = true;
        btnPersonalInfo.disabled = true;
        btnWarranty.disabled = true;
        btnCompany.disabled = true;
    }
    
    /**
     * Hides the loading indicator
     */
    function hideLoading() {
        loadingIndicator.classList.add('d-none');
        sendButton.disabled = false;
        btnPersonalInfo.disabled = false;
        btnWarranty.disabled = false;
        btnCompany.disabled = false;
    }
    
    /**
     * Scrolls the chat container to the bottom
     */
    function scrollToBottom() {
        setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 50);
    }
    
    /**
     * Sends a user message to the server for RAG processing
     * 
     * This function:
     * 1. Adds the user message to the UI
     * 2. Sends the entire conversation history to the FastAPI backend
     * 3. Processes the response from Azure OpenAI enhanced with Azure AI Search results
     * 4. Extracts any citations from the context
     * 5. Handles errors gracefully with user-friendly messages
     */
    function sendMessage(text) {
        hideError();
        
        // Add user message to UI
        addUserMessage(text);
        
        // Clear input field
        chatInput.value = '';
        
        // Add user message to chat history
        const userMessage = {
            role: 'user',
            content: text
        };
        messages.push(userMessage);
        
        // Show loading indicator
        showLoading();
        
        // Send request to server
        fetch('/api/chat/completion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: messages
            })
        })
        .then(response => {
            if (!response.ok) {
                // Try to parse the error response
                return response.json().then(errorData => {
                    throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
                }).catch(e => {
                    // If can't parse JSON, use generic error
                    throw new Error(`HTTP error! Status: ${response.status}`);
                });
            }
            // Log raw response for debugging
            return response.json();
        })
        .then(data => {
            hideLoading();
            
            if (data.error) {
                // Handle API error
                showError(data.message || 'An error occurred');
                return;
            }
            
            const choice = data.choices && data.choices.length > 0 ? data.choices[0] : null;
            if (!choice || !choice.message || !choice.message.content) {
                showError('No answer received from the AI service.');
                return;
            }
            
            // Get message data
            const message = choice.message;
            const content = message.content;
            
            // Extract citations from context
            const citations = message.context?.citations || [];
            
            // Add assistant message to UI
            addAssistantMessage(content, citations);
            
            // Add assistant message to chat history
            const assistantMessage = {
                role: 'assistant',
                content: content
            };
            messages.push(assistantMessage);
        })
        .catch(error => {
            hideLoading();
            showError(`Error: ${error.message}`);
            console.error('Error:', error);
        });
    }
});
