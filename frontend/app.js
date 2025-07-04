// Application state
        let currentUser = null;
        let chatMessages = [];
        let isTyping = false;

        // Mock user database
        const users = {
            'admin': { password: 'admin', name: 'Admin User', email: 'admin@plantchat.com' },
            'john': { password: 'password', name: 'John Doe', email: 'john.doe@example.com' },
            'demo': { password: 'demo', name: 'Demo User', email: 'demo@plantchat.com' },
            'user': { password: '123', name: 'Test User', email: 'test@example.com' }
        };

        // DOM elements
        const loginScreen = document.getElementById('loginScreen');
        const chatScreen = document.getElementById('chatScreen');
        const headerText = document.getElementById('headerText');
        const loginForm = document.getElementById('loginForm');
        const usernameInput = document.getElementById('usernameInput');
        const passwordInput = document.getElementById('passwordInput');
        const signInBtn = document.getElementById('signInBtn');
        const errorMessage = document.getElementById('errorMessage');

        // Chat elements
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const chatMessagesContainer = document.getElementById('chatMessages');
        const welcomeMessage = document.getElementById('welcomeMessage');
        const typingIndicator = document.getElementById('typingIndicator');
        const userAccount = document.getElementById('userAccount');
        const accountDropdown = document.getElementById('accountDropdown');
        const newChatBtn = document.getElementById('newChatBtn');
        const userName = document.getElementById('userName');
        const userAvatar = document.getElementById('userAvatar');

        // Login form handler
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const username = usernameInput.value.trim();
            const password = passwordInput.value.trim();
            
            if (authenticateUser(username, password)) {
                showLoading();
                setTimeout(() => {
                    switchToChat();
                    hideLoading();
                }, 1500); // Simulate login delay
            } else {
                showError();
            }
        });

        // Authentication function
        function authenticateUser(username, password) {
            const user = users[username.toLowerCase()];
            if (user && user.password === password) {
                currentUser = {
                    username: username,
                    name: user.name,
                    email: user.email,
                    avatar: user.name.split(' ').map(n => n[0]).join('').toUpperCase()
                };
                return true;
            }
            return false;
        }

        // UI transition functions
        function switchToChat() {
            // Update header
            headerText.textContent = 'Desktop - 2';
            
            // Hide login screen
            loginScreen.classList.add('fade-out');
            
            setTimeout(() => {
                loginScreen.style.display = 'none';
                chatScreen.style.display = 'flex';
                chatScreen.classList.add('fade-in');
                
                // Update user display
                updateUserDisplay();
                
                // Focus on chat input
                chatInput.focus();
            }, 300);
        }

        function switchToLogin() {
            // Update header
            headerText.textContent = 'Login Screen';
            
            // Hide chat screen
            chatScreen.classList.remove('fade-in');
            chatScreen.classList.add('fade-out');
            
            setTimeout(() => {
                chatScreen.style.display = 'none';
                loginScreen.style.display = 'flex';
                loginScreen.classList.remove('fade-out');
                
                // Reset login form
                loginForm.reset();
                errorMessage.style.display = 'none';
                
                // Reset chat state
                resetChat();
                
                // Clear current user
                currentUser = null;
            }, 300);
        }

        function showLoading() {
            signInBtn.classList.add('loading');
            signInBtn.textContent = 'Signing In...';
            signInBtn.disabled = true;
        }

        function hideLoading() {
            signInBtn.classList.remove('loading');
            signInBtn.textContent = 'Sign In';
            signInBtn.disabled = false;
        }

        function showError() {
            errorMessage.style.display = 'block';
            usernameInput.style.borderColor = '#e74c3c';
            passwordInput.style.borderColor = '#e74c3c';
            
            setTimeout(() => {
                errorMessage.style.display = 'none';
                usernameInput.style.borderColor = '';
                passwordInput.style.borderColor = '';
            }, 3000);
        }

        // Chat functionality
        function updateUserDisplay() {
            if (currentUser) {
                userName.textContent = currentUser.name;
                userAvatar.textContent = currentUser.avatar;
            }
        }

        // Chat input handlers
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            
            sendBtn.disabled = this.value.trim() === '';
        });

        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendBtn.addEventListener('click', sendMessage);
        newChatBtn.addEventListener('click', startNewChat);

        // User account dropdown
        userAccount.addEventListener('click', function(e) {
            e.stopPropagation();
            accountDropdown.classList.toggle('show');
        });

        document.addEventListener('click', function() {
            accountDropdown.classList.remove('show');
        });

        // Chat functions
        function sendMessage() {
            const message = chatInput.value.trim();
            if (message === '' || isTyping) return;

            // Add user message
            addMessage(message, 'user');
            chatInput.value = '';
            chatInput.style.height = 'auto';
            sendBtn.disabled = true;

            // Show chat area if first message
            if (chatMessages.length === 1) {
                welcomeMessage.style.display = 'none';
                chatMessagesContainer.style.display = 'block';
            }

            // Simulate AI response
            simulateAIResponse(message);
        }

        function addMessage(content, sender, parseMarkdown = false) {
    const message = {
        content: content,
        sender: sender,
        timestamp: new Date()
    };
    
    chatMessages.push(message);
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (parseMarkdown && sender === 'assistant') {
        messageContent.innerHTML = marked.parse(content);
    } else {
        messageContent.textContent = content;
    }
    
    messageElement.appendChild(messageContent);
    
    // Insert before typing indicator
    chatMessagesContainer.insertBefore(messageElement, typingIndicator);
    
    // Scroll to bottom
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
}

//         function simulateAIResponse(userMessage) {
//     isTyping = true;
//     typingIndicator.style.display = 'block';
//     chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;

//     // Simulate thinking time
//     setTimeout(() => {
//         typingIndicator.style.display = 'none';
        
//         // Generate a response
//         let response = generateResponse(userMessage);
//         addMessage(response, 'assistant', true); // Add the parseMarkdown flag
        
//         isTyping = false;
//     }, 1500 + Math.random() * 1500);
// }

function simulateAIResponse(userMessage) {
    isTyping = true;
    typingIndicator.style.display = 'block';
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;

    fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: userMessage,
            username: currentUser?.username || "guest"  // Use current username
        })
    })    
    .then(res => res.json())
    .then(data => {
        typingIndicator.style.display = 'none';
        addMessage(data.response, 'assistant', true);
        isTyping = false;
    })
    .catch(err => {
        typingIndicator.style.display = 'none';
        addMessage("Error: " + err.message, 'assistant');
        isTyping = false;
    });
}

        function generateResponse(message) {
    const responses = [
        `**Hello ${currentUser.name}!** That's an interesting question. Let me help you with that.

Here are some key points to consider:
- This is a *sample response* with markdown
- I can format text with **bold** and *italic*
- Code snippets look like this: \`console.log('hello')\`

> *Note*: This is a test response to demonstrate markdown formatting.`,

        `I understand what you're asking. Here's what I think...

## My Analysis

**Key points:**
1. **First point**: This demonstrates numbered lists
2. **Second point**: With some *emphasis* added  
3. **Third point**: Including \`inline code\` examples

*Hope this helps!*`,

        `Great question! Based on what you've shared, I'd suggest:

### Recommendations

- **Option 1**: Try this approach first
- **Option 2**: Consider this alternative
- **Option 3**: As a backup plan

> **Important**: These are general suggestions based on your query.`,

        `I'm here to help! Let me provide you with some information about that.

**Quick Summary:**
> This is a blockquote with important information

*Detailed breakdown:*
- Point one with *emphasis*
- Point two with \`code example\`  
- Point three with **strong importance**

Feel free to ask for clarification!`,

        `Thanks for asking! Here's my perspective on this topic...

## Analysis Results

**Pros:**
- Advantage one
- Advantage two
- Advantage three

**Cons:**  
- Consideration one
- Consideration two

*Overall recommendation:* Based on the above factors, I suggest proceeding with caution.`
    ];

    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
        return `# Hello ${currentUser.name}! 👋

It's **great** to meet you! How can I assist you today?

## I can help with:
- **Technical questions** - Code, programming, etc.
- **General information** - Facts, explanations  
- **Recommendations** - Suggestions and advice

> *Tip*: Be specific in your questions for better responses!`;

    } else if (lowerMessage.includes('help')) {
        return `## I'm here to help you, ${currentUser.name}! 

Feel free to ask me anything you'd like to know.

### Popular topics:
- **Technical questions** - Code, programming, etc.
- **General information** - Facts, explanations
- **Recommendations** - Suggestions and advice

> *Remember*: The more specific your question, the better I can help!`;

    } else if (lowerMessage.includes('weather')) {
        return `## Weather Information 🌤️

I'd love to help with weather information, but I don't have access to *real-time weather data*.

**Recommended alternatives:**
- Weather apps on your phone
- \`weather.com\` or \`weather.gov\`
- Local news websites
- Voice assistants like Siri or Google

> *Sorry I can't provide current conditions directly!*`;

    } else if (lowerMessage.includes('time')) {
        return `## Current Time ⏰

**The current time is:** \`${new Date().toLocaleTimeString()}\`

*Additional time info:*
- **Date**: ${new Date().toLocaleDateString()}
- **Day**: ${new Date().toLocaleDateString('en-US', { weekday: 'long' })}

> Time is based on your local system settings.`;

    } else if (lowerMessage.includes('name')) {
        return `## Your Profile Information 👤

**Name:** ${currentUser.name}  
**Username:** \`${currentUser.username}\`

### Account Status:
- Currently logged in ✅
- Session active
- Ready to assist!

> Need to update your profile? Let me know how I can help!`;

    } else if (lowerMessage.includes('test') || lowerMessage.includes('markdown')) {
        return `# Markdown Test Response

This response demonstrates various **markdown features**:

## Headers
### Like this sub-header

## Text Formatting
- **Bold text** for emphasis
- *Italic text* for subtle emphasis
- \`Inline code\` for technical terms

## Lists
1. **Numbered lists** work great
2. *Perfect for* step-by-step instructions
3. Easy to follow

**Bullet points:**
- Item one
- Item two with \`code\`
- Item three with **bold**

## Blockquotes
> This is a blockquote that stands out from regular text
> 
> Perfect for important notes or quotes

## Code Examples
Here's some code: \`const message = "Hello World!"\`

*This demonstrates how your LLM responses will look!*`;

    } else {
        return responses[Math.floor(Math.random() * responses.length)];
    }
}

        function startNewChat() {
            resetChat();
        }

        function resetChat() {
            // Clear chat messages
            chatMessages = [];
            chatMessagesContainer.style.display = 'none';
            welcomeMessage.style.display = 'block';
            
            // Clear chat UI
            const messageElements = chatMessagesContainer.querySelectorAll('.message');
            messageElements.forEach(el => el.remove());
            
            // Reset input
            chatInput.value = '';
            chatInput.style.height = 'auto';
            sendBtn.disabled = true;
            
            // Hide typing indicator
            typingIndicator.style.display = 'none';
            isTyping = false;
        }

        // Account dropdown functions
        function editProfile() {
            const newName = prompt('Enter new name:', currentUser.name);
            if (newName && newName.trim()) {
                currentUser.name = newName.trim();
                currentUser.avatar = newName.trim().split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
                updateUserDisplay();
            }
            accountDropdown.classList.remove('show');
        }

        function changeSettings() {
            alert('Settings panel would open here. This is a demo interface.');
            accountDropdown.classList.remove('show');
        }

        function logout() {
            if (confirm('Are you sure you want to logout?')) {
                switchToLogin();
            }
            accountDropdown.classList.remove('show');
        }

        // Initialize app
        document.addEventListener('DOMContentLoaded', function() {
            // Focus on username input
            usernameInput.focus();
            
            // Add demo user info
            const demoInfo = document.createElement('div');
            demoInfo.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 20px;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-size: 12px;
                z-index: 1000;
            `;
            demoInfo.innerHTML = `
                <strong>Demo Login:</strong><br>
                Username: demo<br>
                Password: demo
            `;
            document.body.appendChild(demoInfo);
        });