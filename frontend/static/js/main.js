// ================================
// COSMIC NEO RAG - MAIN JAVASCRIPT
// ================================

// Cosmic Particle System
function createCosmicParticleSystem() {
    const container = document.getElementById('particleSystem');
    if (!container) return;
    
    const particleCount = 25;
    
    for (let i = 0; i < particleCount; i++) {
        createCosmicParticle(container);
    }
}

function createCosmicParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    // Random particle type
    const types = ['gold', 'orange', 'white'];
    const type = types[Math.floor(Math.random() * types.length)];
    particle.classList.add(type);
    
    // Random properties
    const size = Math.random() * 3 + 1.5;
    const startX = Math.random() * window.innerWidth;
    const duration = Math.random() * 20 + 20;
    const delay = Math.random() * 10;
    
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    particle.style.left = startX + 'px';
    particle.style.top = '100vh';
    particle.style.animationDuration = duration + 's';
    particle.style.animationDelay = delay + 's';
    particle.style.animation = `cosmicParticleRise ${duration}s linear ${delay}s infinite`;
    
    container.appendChild(particle);
    
    // Remove and recreate particle after animation
    setTimeout(() => {
        if (particle.parentNode) {
            particle.remove();
            createCosmicParticle(container);
        }
    }, (duration + delay) * 1000);
}

// Add particle animations
function addCosmicParticleAnimations() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes cosmicParticleRise {
            0% { 
                transform: translateY(0) scale(0);
                opacity: 0;
            }
            20% {
                opacity: 0.8;
                transform: scale(1);
            }
            80% {
                opacity: 0.8;
            }
            100% { 
                transform: translateY(-100vh) translateX(${Math.random() * 80 - 40}px) scale(0.5);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

// Enhanced Chat Bot Class
class CosmicLegalChatBot {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        
        this.isTyping = false;
        this.init();
    }
    
    init() {
        // Event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Input hints click handlers
        document.querySelectorAll('.hint-item').forEach(hint => {
            hint.addEventListener('click', () => {
                this.messageInput.value = hint.textContent.replace(/^[^\s]+ /, '');
                this.messageInput.focus();
            });
        });
        
        // Focus on input
        this.messageInput.focus();
        
        // Hide loading overlay after init
        setTimeout(() => {
            if (this.loadingOverlay) {
                this.loadingOverlay.style.display = 'none';
            }
        }, 1000);
    }
    
    addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        // Add typing effect for bot messages
        if (!isUser) {
            this.typewriterEffect(bubbleDiv, content);
        } else {
            bubbleDiv.textContent = content;
        }
        
        messageDiv.appendChild(bubbleDiv);
        this.chatMessages.appendChild(messageDiv);
        
        this.scrollToBottom();
    }
    
    typewriterEffect(element, text) {
        let i = 0;
        const speed = 30;
        
        function typeWriter() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, speed);
            }
        }
        
        typeWriter();
    }
    
    showTypingIndicator() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage(message, true);
        this.messageInput.value = '';
        this.showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // Simulate thinking time
            setTimeout(() => {
                this.hideTypingIndicator();
                this.addMessage(data.response || 'Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể hỏi lại được không?');
            }, 1500 + Math.random() * 1000);

        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('Xin lỗi, có sự cố trong hệ thống. Vui lòng thử lại sau!');
        }
    }
}

// Cosmic Animations and Effects
function initCosmicAnimations() {
    // Entrance animation for main container
    const mainContainer = document.querySelector('.main-container');
    if (mainContainer) {
        mainContainer.style.opacity = '0';
        mainContainer.style.transform = 'translateY(50px)';
        
        setTimeout(() => {
            mainContainer.style.transition = 'all 1s ease-out';
            mainContainer.style.opacity = '1';
            mainContainer.style.transform = 'translateY(0)';
        }, 300);
    }
    
    // Intersection Observer for animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
            }
        });
    }, { threshold: 0.2 });
    
    // Observe elements for animations
    document.querySelectorAll('.welcome-card, .feature-list li').forEach(el => {
        observer.observe(el);
    });
}

// Interactive Effects
function addCosmicInteractiveEffects() {
    const sendButton = document.getElementById('sendButton');
    const messageInput = document.getElementById('messageInput');
    
    // Send button effects
    if (sendButton) {
        sendButton.addEventListener('click', () => {
            sendButton.style.transform = 'scale(0.95)';
            setTimeout(() => {
                sendButton.style.transform = '';
            }, 150);
        });
        
        sendButton.addEventListener('mouseenter', () => {
            sendButton.style.boxShadow = '0 0 40px rgba(255, 215, 0, 0.4)';
        });
        
        sendButton.addEventListener('mouseleave', () => {
            sendButton.style.boxShadow = '';
        });
    }
    
    // Input focus effects
    if (messageInput) {
        messageInput.addEventListener('focus', () => {
            const inputGroup = messageInput.closest('.input-group');
            if (inputGroup) {
                inputGroup.style.transform = 'translateY(-2px)';
                inputGroup.style.boxShadow = '0 0 30px rgba(255, 215, 0, 0.2)';
            }
        });
        
        messageInput.addEventListener('blur', () => {
            const inputGroup = messageInput.closest('.input-group');
            if (inputGroup) {
                inputGroup.style.transform = '';
                inputGroup.style.boxShadow = '';
            }
        });
    }
    
    // Hint items hover effects
    document.querySelectorAll('.hint-item').forEach(hint => {
        hint.addEventListener('mouseenter', () => {
            hint.style.boxShadow = '0 4px 15px rgba(255, 215, 0, 0.2)';
        });
        
        hint.addEventListener('mouseleave', () => {
            hint.style.boxShadow = '';
        });
    });
    
    // Feature list hover effects
    document.querySelectorAll('.feature-list li').forEach(item => {
        item.addEventListener('mouseenter', () => {
            item.style.boxShadow = '0 8px 25px rgba(255, 215, 0, 0.15)';
        });
        
        item.addEventListener('mouseleave', () => {
            item.style.boxShadow = '';
        });
    });
}

// Cosmic Background Animation Control
function controlCosmicBackground() {
    const cosmicBackground = document.querySelector('.cosmic-background');
    
    if (cosmicBackground) {
        // Adjust animation based on user preference
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            cosmicBackground.style.animation = 'none';
            document.querySelectorAll('.gradient-orb').forEach(orb => {
                orb.style.animation = 'none';
            });
        }
        
        // Performance optimization
        let animationId;
        function optimizeBackground() {
            if (document.hidden) {
                // Pause animations when tab is not visible
                cosmicBackground.style.animationPlayState = 'paused';
            } else {
                cosmicBackground.style.animationPlayState = 'running';
            }
        }
        
        document.addEventListener('visibilitychange', optimizeBackground);
    }
}

// Responsive Particle System
function handleResponsiveParticles() {
    function adjustParticleCount() {
        const container = document.getElementById('particleSystem');
        if (!container) return;
        
        const isMobile = window.innerWidth < 768;
        const isTablet = window.innerWidth < 1024;
        
        let targetCount = 25;
        if (isMobile) targetCount = 10;
        else if (isTablet) targetCount = 15;
        
        const currentCount = container.children.length;
        const diff = targetCount - currentCount;
        
        if (diff > 0) {
            for (let i = 0; i < diff; i++) {
                createCosmicParticle(container);
            }
        } else if (diff < 0) {
            for (let i = 0; i < Math.abs(diff); i++) {
                if (container.firstChild) {
                    container.removeChild(container.firstChild);
                }
            }
        }
    }
    
    // Debounced resize handler
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(adjustParticleCount, 300);
    });
}

// Keyboard Shortcuts
function addKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Alt + / to focus input
        if (e.altKey && e.key === '/') {
            e.preventDefault();
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.focus();
            }
        }
        
        // Escape to clear input
        if (e.key === 'Escape') {
            const messageInput = document.getElementById('messageInput');
            if (messageInput && document.activeElement === messageInput) {
                messageInput.value = '';
                messageInput.blur();
            }
        }
    });
}

// Error Handling
function setupErrorHandling() {
    window.addEventListener('error', (e) => {
        console.error('Cosmic Interface Error:', e.error);
        
        // Show user-friendly error if chat fails
        const chatBot = window.cosmicChatBot;
        if (chatBot && e.error.message.includes('fetch')) {
            chatBot.hideTypingIndicator();
            chatBot.addMessage('🌌 Kết nối vũ trụ bị gián đoạn. Vui lòng thử lại!');
        }
    });
    
    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (e) => {
        console.error('Unhandled Promise Rejection:', e.reason);
    });
}

// Performance Monitor
function setupPerformanceMonitoring() {
    if ('performance' in window) {
        window.addEventListener('load', () => {
            setTimeout(() => {
                const perfData = performance.getEntriesByType('navigation')[0];
                if (perfData && perfData.loadEventEnd > 3000) {
                    console.warn('Slow page load detected:', perfData.loadEventEnd + 'ms');
                    // Could reduce particle count or effects here
                }
            }, 1000);
        });
    }
}

// Main Initialization
function initializeCosmicInterface() {
    // Core systems
    addCosmicParticleAnimations();
    createCosmicParticleSystem();
    
    // Chat system
    window.cosmicChatBot = new CosmicLegalChatBot();
    
    // Visual effects
    initCosmicAnimations();
    addCosmicInteractiveEffects();
    controlCosmicBackground();
    
    // Responsive and accessibility
    handleResponsiveParticles();
    addKeyboardShortcuts();
    
    // Error handling and monitoring
    setupErrorHandling();
    setupPerformanceMonitoring();
    
    // Mark as loaded
    document.body.classList.add('cosmic-loaded');
    
    console.log('🌌 NEO RAG Cosmic Interface Initialized');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeCosmicInterface);
} else {
    initializeCosmicInterface();
}

// Export for external use
window.CosmicNEORAG = {
    chatBot: () => window.cosmicChatBot,
    createParticle: createCosmicParticle,
    addMessage: (content, isUser) => window.cosmicChatBot?.addMessage(content, isUser)
}; 