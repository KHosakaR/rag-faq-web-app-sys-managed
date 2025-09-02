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

    // Stream Settings
    const USE_STREAM = true; // ストリーミングを優先
    
    // Quick response buttons
    // const btnPersonalInfo = document.getElementById('btn-personal-info');
    // const btnWarranty = document.getElementById('btn-warranty');
    // const btnCompany = document.getElementById('btn-company');
    
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
    // btnPersonalInfo.addEventListener('click', () => sendQuickQuestion("What does Contoso do with my personal information?"));
    // btnWarranty.addEventListener('click', () => sendQuickQuestion("How do I file a warranty claim?"));
    // btnCompany.addEventListener('click', () => sendQuickQuestion("Tell me about your company."));

    // 小さなHTMLエスケープ
    function escHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    // セッションID（ローカルに保持して継続文脈に使う）
    function getSessionId() {
    let sid = localStorage.getItem('sessionId');
    if (!sid) {
        sid = (window.crypto?.randomUUID?.() || (String(Date.now()) + Math.random().toString(16).slice(2)));
        localStorage.setItem('sessionId', sid);
    }
    return sid;
    }
    
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

            // API側のsources_textをリンク化（[1] https://example.com）
            formattedContent = formattedContent.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
            
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
            
            // 「引用元：」の一覧を末尾に差し込む（citations があれば）
            if (citations && citations.length > 0) {
                let listHtml = '\n\n<div class="source-list mt-2"><strong>引用元：</strong><br>';
                citations.forEach((c, i) => {
                    const url = c.url || c.filePath || '';
                    if (!url) return;
                    const idx = i + 1;
                    // [docN] が本文に無くても参照できるよう、マッピングを用意
                    if (!messageCitations[idx]) {
                        messageCitations[idx] = JSON.stringify({
                            title: c.title || url,
                            content: c.content || '',
                            filePath: c.filePath || url,
                            url
                        });
                    }
                    listHtml += `[${idx}] <a class="cite-link" style="cursor:pointer;" data-message-id="${messageId}" data-index="${idx}">${escHtml(url)}</a><br>`;
                });
                listHtml += '</div>';
                formattedContent += listHtml;
            }

            // **MarkdownをHTMLに変換**
            messageContent.innerHTML = marked.parse(formattedContent);
            
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
                        // showCitationModal(citationData);
                        openCitation(citationData);
                    });
                });

                // 「引用元：」側のリンクも同じ動きにする
                const citeLinks = messageContent.querySelectorAll('.cite-link[data-index]');
                citeLinks.forEach(link => {
                    link.addEventListener('click', function(e){
                        e.preventDefault();
                        e.stopPropagation();
                        const messageId = this.getAttribute('data-message-id');
                        const idx = this.getAttribute('data-index');
                        const messageElement = document.getElementById(messageId);
                        const stored = JSON.parse(messageElement.getAttribute('data-citations') || '{}');
                        const citationData = JSON.parse(stored[idx]);
                        openCitation(citationData);
                    });
                });
            }, 100);
        }
        
        chatHistory.appendChild(messageNode);
        scrollToBottom();
    }

    // ---- ストリーミング用ヘルパ ----
    function startAssistantStreamBubble() {
        // 空のアシスタントバブルを作成（本文は後で逐次追記）
        const node = assistantMessageTemplate.content.cloneNode(true);
        const card = node.querySelector('.card');
        const contentEl = node.querySelector('.message-content');
        const messageId = 'msg-' + Date.now();
        card.setAttribute('id', messageId);
        // streaming 中はまずプレーンテキストで高速表示
        contentEl.textContent = '';
        chatHistory.appendChild(node);
        scrollToBottom();
        return { cardEl: document.getElementById(messageId), contentEl: document.querySelector('#' + messageId + ' .message-content'), messageId };
    }

    function appendStreamToken(state, delta) {
        // 逐次追記：一旦 textContent に追加（高速）
        state.buffer += delta;
        state.contentEl.textContent = state.buffer;
        scrollToBottom();
    }

    function finalizeStreamBubble(state, citations) {
        // 最後に Markdown 変換＆出典バッジ化
        let formatted = state.buffer.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
        // [docN] をバッジに差し替え
        const pattern = /\[doc(\d+)\]/g;
        const messageCitations = {};
        formatted = formatted.replace(pattern, (match, index) => {
            const idx = parseInt(index);
            if (idx > 0 && idx <= (citations?.length || 0)) {
                const c = citations[idx - 1] || {};
                const citationData = JSON.stringify({
                    title: c.title || '',
                    content: c.content || '',
                    filePath: c.filePath || '',
                    url: c.url || ''
                });
                messageCitations[idx] = citationData;
                return `<a class="badge bg-primary rounded-pill" style="cursor: pointer;" data-message-id="${state.messageId}" data-index="${idx}">${idx}</a>`;
            }
            return match;
        });

        // 末尾に「引用元：」を追加
        if (citations && citations.length > 0) {
            let listHtml = '\n\n<div class="source-list mt-2"><strong>引用元：</strong><br>';
            citations.forEach((c, i) => {
                const url = c.url || c.filePath || '';
                if (!url) return;
                const idx = i + 1;
                // [docN] が本文に無くても参照できるよう、マッピングを用意
                if (!messageCitations[idx]) {
                    messageCitations[idx] = JSON.stringify({
                        title: c.title || url,
                        content: c.content || '',
                        filePath: c.filePath || url,
                        url
                    });
                }
                listHtml += `[${idx}] <a class="cite-link" style="cursor:pointer;" data-message-id="${state.messageId}" data-index="${idx}">${escHtml(url)}</a><br>`;
            });
            listHtml += '</div>';
            formatted += listHtml;
        }

        state.contentEl.innerHTML = marked.parse(formatted);
        state.cardEl.setAttribute('data-citations', JSON.stringify(messageCitations));

        // バッジのクリックリスナー
        const badges = state.contentEl.querySelectorAll('.badge[data-index]');
        badges.forEach(badge => {
            badge.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                const messageId = this.getAttribute('data-message-id');
                const idx = this.getAttribute('data-index');
                const messageElement = document.getElementById(messageId);
                const stored = JSON.parse(messageElement.getAttribute('data-citations') || '{}');
                const citationData = JSON.parse(stored[idx]);
                openCitation(citationData);
            });
        });

        // 「引用元：」側のリンク
        const citeLinks = state.contentEl.querySelectorAll('.cite-link[data-index]');
        citeLinks.forEach(link => {
            link.addEventListener('click', function(e){
                e.preventDefault();
                e.stopPropagation();
                const messageId = this.getAttribute('data-message-id');
                const idx = this.getAttribute('data-index');
                const messageElement = document.getElementById(messageId);
                const stored = JSON.parse(messageElement.getAttribute('data-citations') || '{}');
                const citationData = JSON.parse(stored[idx]);
                openCitation(citationData);
            });
        });
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

    function openCitation(cite) {
        const raw = cite.url || cite.filePath;
        if (!raw) {
            alert("Citation URL not found");
            return;
        }

        // 小ヘルパ
        const isHttpUrl = (s) => {
            try { const u = new URL(s); return /^https?:/i.test(u.protocol); } catch { return false; }
        };
        const isInternalBlob = (s) => {
            if (!window.BLOB_ACCOUNT_HOST) return false;
            try { return new URL(s).host === window.BLOB_ACCOUNT_HOST; } catch { return false; }
        };

        // すでにSAS付き（sig= 含む） or 外部サイト → そのまま開く
        if (isHttpUrl(raw) && (raw.includes("sig=") || !isInternalBlob(raw))) {
            window.open(raw, "_blank", "noopener");
            return;
        }

        // ここまで来たら自アカウントの Blob（URL でも 'container/blob' でも可）→ SAS を発行して開く
        fetch(`/api/blob/sas?path=${encodeURIComponent(raw)}`)
            .then(async r => {
            if (!r.ok) {
                const detail = await r.json().catch(() => ({}));
                if (r.status === 403) throw new Error(detail?.detail || "許可されていない参照です。");
                throw new Error(detail?.detail || "SAS発行に失敗しました。");
            }
            return r.json();
            })
            .then(data => window.open(data.url, "_blank", "noopener"))
            .catch(err => {
            console.error(err);
            alert("SAS URL を取得できませんでした");
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
        // btnPersonalInfo.disabled = true;
        // btnWarranty.disabled = true;
        // btnCompany.disabled = true;
    }
    
    /**
     * Hides the loading indicator
     */
    function hideLoading() {
        loadingIndicator.classList.add('d-none');
        sendButton.disabled = false;
        // btnPersonalInfo.disabled = false;
        // btnWarranty.disabled = false;
        // btnCompany.disabled = false;
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
        // ストリーミング優先
        if (USE_STREAM) {
            sendMessageStreaming(messages)
            .catch(err => {
                console.warn("stream failed, fallback to non-stream:", err);
                // フォールバック：従来エンドポイントで一括取得
                return sendMessageNonStream(messages);
               });
            return;
        }
        // 非ストリームのまま
        sendMessageNonStream(messages);
    }

    // --- 非ストリーム（従来） ---
    function sendMessageNonStream(history) {
        // Send request to server (non-stream)
        return fetch('/api/chat/completion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-Id': getSessionId(),
            },
            body: JSON.stringify({
                messages: history
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

    // --- ストリーミング ---
    async function sendMessageStreaming(history) {
        try {
            const res = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-Id': getSessionId(),
                },
                body: JSON.stringify({ history })
            });

            if (!res.ok || !res.body) {
                throw new Error(`stream http ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            const state = startAssistantStreamBubble();
            state.buffer = '';

            let citations = [];

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                let idx;
                while ((idx = buffer.indexOf('\n\n')) >= 0) {
                    const raw = buffer.slice(0, idx).trim();
                    buffer = buffer.slice(idx + 2);
                    if (!raw) continue;

                    // SSE の event/data を読む
                    const lines = raw.split('\n');
                    let event = 'message';
                    let data = null;
                    for (const line of lines) {
                        if (line.startsWith('event:')) event = line.slice(6).trim();
                        if (line.startsWith('data:')) {
                            try { data = JSON.parse(line.slice(5).trim()); } catch {}
                        }
                    }
                    if (!data) continue;

                    if (event === 'token') {
                        appendStreamToken(state, data.delta || '');
                    } else if (event === 'citations') {
                        citations = data.items || [];
                    } else if (event === 'done') {
                        const isLow = !!data.is_low;
                        if (isLow) {
                            // 低信頼：これまでのRAGトークンをUIから消去して、以降のWeb検索トークンを同じバブルで受ける
                            state.buffer = '';
                            state.contentEl.textContent = '';
                            // finalize も履歴追加も行わず、そのまま次の token（案内文→Web検索）を待つ
                            continue;
                        } else {
                            finalizeStreamBubble(state, citations);
                            hideLoading();
                            // 会話履歴にも最終テキストを積む
                            messages.push({ role: 'assistant', content: state.buffer });
                        }
                    } else if (event === 'error') {
                        throw new Error(data.message || 'stream error');
                    }
                }
            }
        } catch (err) {
            hideLoading();
            showError(`Stream Error: ${err.message}`);
            throw err;
        }
    }
});
