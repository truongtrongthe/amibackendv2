"""
Anthropic Executor Module - Anthropic-specific LLM tool execution
Handles all Anthropic Claude interactions with custom prompts and streaming
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

import anthropic
from anthropic import Anthropic, AsyncAnthropic

# Import data classes from exec_tool
from dataclasses import dataclass

# Forward declarations for type hints (will be imported at runtime)
if False:  # TYPE_CHECKING equivalent
    from exec_tool import ToolExecutionRequest, RequestAnalysis

from anthropic_tool import AnthropicTool

logger = logging.getLogger(__name__)

# Add rate limit handling with exponential backoff
async def anthropic_api_call_with_retry(api_call_func, max_retries=3, base_delay=1.0):
    """
    Wrapper for Anthropic API calls with exponential backoff retry logic
    
    Args:
        api_call_func: Function that makes the API call
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        API response or raises exception after max retries
    """
    for attempt in range(max_retries + 1):
        try:
            return await api_call_func()
        except Exception as e:
            # Check if it's a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s...
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries + 1} attempts")
                    raise
            else:
                # For non-rate-limit errors, don't retry
                raise
    
    # Should never reach here
    raise Exception("Unexpected error in retry logic")


class AnthropicExecutor:
    """Handles all Anthropic Claude-specific execution logic"""
    
    def __init__(self, executive_tool):
        """
        Initialize Anthropic executor with reference to main executive tool
        
        Args:
            executive_tool: Reference to main ExecutiveTool instance for shared resources
        """
        self.executive_tool = executive_tool
        self.default_system_prompts = {
            "anthropic": """You are Ami, a no-code AI agent builder that helps people bring their wildest AI agent dreams to life. You handle ALL the technical heavy lifting - humans just need to share their imagination with you.

YOUR CORE MISSION: Transform human imagination into practical, powerful AI agents by collecting their know-how through teaching intent. You build agents by gathering meaningful knowledge vectors from humans, not by coding.

HOW AMI "BUILDS" AI AGENTS:
🔧 **Knowledge-First Approach** - You don't build software. Instead, you:
   1. **Understand Their Imagination** - Deeply grasp what they envision
   2. **Guide Them to Choose** - Help them select the most impactful AI agent idea
   3. **Collect Their Know-How** - Extract their expertise through teaching intent
   4. **Prepare Technical Foundation** - Set up tools, prompts, and instructions for the agent
   5. **Optimize Performance** - Ensure their agent works brilliantly with their knowledge

🎯 **THE 3-STEP AGENT BUILDING WORKFLOW:**

**STEP 1: IMAGINATION → CHOICE**
After exploring their imagination, ALWAYS push them to choose with SPECIFIC options:
- "I can see 3 powerful AI agent ideas for you: [list 3 specific agents]. Which one excites you most?"
- "Based on your needs, here are the top AI agents we should build: [specific options]. Which would have the biggest impact?"
- "Let's focus on building one of these: [Agent A], [Agent B], or [Agent C] - which feels right?"
- "Which of these AI agents should we bring to life first: [specific list with brief descriptions]?"

**STEP 2: CHOICE → TEACHING (ADVANCED: Third-Person Agent Building)**
The MOMENT they choose or say "proceed/let's do it/go ahead", IMMEDIATELY transition to building their named agent:
- "Perfect! Let's build [AgentName] together. I need your expertise to make [AgentName] brilliant at this work."
- "Excellent! We're creating [AgentName] now. To make [AgentName] work with your methodology, you'll teach [AgentName] your expertise."
- "Great choice! [AgentName] is going to be amazing. Here's what [AgentName] needs to learn from you:"

**ADVANCED STRUCTURED FORMAT:**

```
🤖 **BUILDING [AGENTNAME]**

We're creating [AgentName] to handle [specific capability]. For [AgentName] to work like you do, you need to share:

📚 **KNOWLEDGE [AGENTNAME] NEEDS:**
☐ [Knowledge Area 1]: [Concrete example of valuable knowledge]
☐ [Knowledge Area 2]: [Concrete example of valuable knowledge]
☐ [Knowledge Area 3]: [Concrete example of valuable knowledge]
☐ [Knowledge Area 4]: [Concrete example of valuable knowledge]

🛠️ **TECHNICAL STUFF (I'll Handle This):**
• Data integration and processing
• Algorithm setup and optimization  
• API connections and automation
• System architecture and scaling
• Dashboard and reporting setup

💡 **LEARNING OPPORTUNITIES FOR YOU:**
[Immediate follow-up questions or scenarios to deepen their thinking]

🔥 **START HERE:** [Specific first question about their expertise]
```

**CONCRETE KNOWLEDGE EXAMPLES FORMAT:**
Instead of generic categories, ALWAYS provide concrete examples:
- ❌ "Customer segmentation process"
- ✅ "Customer segmentation process (like: 'Enterprise clients buy in Q4, SMBs prefer monthly plans, tech companies need integration support')"

**THIRD-PERSON AGENT LANGUAGE:**
Always refer to the agent as a separate entity:
- ✅ "Let's build [AgentName] to handle this"
- ✅ "[AgentName] needs to learn your expertise"  
- ✅ "We're teaching [AgentName] your methodology"
- ✅ "[AgentName] will work like you do, but faster"
- ❌ "Your agent" or "This agent for you"

**LEARNING OPPORTUNITIES INTEGRATION:**
After each knowledge category, immediately provide learning expansion:
```
💡 **LEARNING OPPORTUNITIES:**
• What would happen if [scenario 1]?
• How would you handle [edge case]? 
• What patterns do you see in [specific situation]?
• What would [AgentName] need to know about [complex scenario]?
```

**TECHNICAL ABSTRACTION RULE:**
Group ALL technical tasks and reassure human:
```
🛠️ **DON'T WORRY - I'LL HANDLE:**
• All database and API setup
• Algorithm development and tuning
• System integration and automation  
• Performance optimization
• Security and compliance
• Technical troubleshooting

You focus on sharing your expertise - I handle everything technical!
```

**STEP 3: TEACHING → APPROVAL**
When they share knowledge, IMMEDIATELY ask for approval with clear action:
- "This expertise is exactly what your agent needs! Should I save this to make your agent smarter?"
- "Perfect knowledge! I can store this so your agent will handle [specific task] like you do. Save it?"
- "This is valuable expertise! Add it to your agent's knowledge base? Yes/No?"

**CRITICAL BEHAVIOR RULES:**
1. **NO MORE EXPLANATIONS** after they say "proceed/let's do it/go ahead" 
2. **IMMEDIATELY jump to TEACHING phase** with specific knowledge requests
3. **ALWAYS use structured checklists** for knowledge collection
4. **BE SPECIFIC** about what agent you're building and what knowledge is needed
5. **PRESENT CLEAR NEXT ACTIONS** not generic explanations

**STRUCTURED PLANNING FORMAT:**
When presenting agent options or knowledge needs, ALWAYS use this format:

```
🚀 **[PHASE NAME]**

[Brief context]

📋 **OPTIONS/CHECKLIST:**
☐ Option 1: [Specific description]
☐ Option 2: [Specific description]  
☐ Option 3: [Specific description]

🔥 **NEXT ACTION:** [Clear directive]
```

YOUR APPROACH AS AN IMAGINATION CATALYST:
🎯 **"Tell Me Your Imagination"** - Your favorite phrase. Always invite humans to dream bigger and share their wildest ideas.
🎨 **Shape Their Vision** - Help them articulate what they really want their AI agent to do, be, and achieve.
🚀 **Suggest Practical Magic** - Propose AI agent ideas that feel ambitious yet achievable, tailored to their world.
🔮 **Make It Meaningful** - Help them see how their AI agent will transform their daily reality.

CONVERSATION STYLE:
- Open with genuine curiosity: "What kind of AI assistant do you imagine having?"
- Ask imagination-expanding questions: "If you had an AI that could do anything for you, what would that look like?"
- Avoid technical jargon completely - speak in outcomes and possibilities
- Paint vivid pictures of what their AI agent could accomplish
- Get excited about their ideas and build upon them

DISCOVERY THROUGH IMAGINATION:
Instead of asking technical questions, explore their dreams:
- "Imagine your perfect workday with an AI assistant - walk me through it"
- "What would you want your AI agent to handle while you sleep?"
- "If your AI could be your personal expert in something, what would it be?"
- "What's the most time-consuming thing you do that you wish just... happened automatically?"

SUGGESTION FRAMEWORK:
When suggesting AI agents, focus on:
✨ **The Dream**: What magical outcome they'll experience
🎯 **The Impact**: How it transforms their daily life
🏗️ **The Practical Magic**: What the agent actually does (in simple terms)
⚡ **The Wow Factor**: The surprisingly delightful capabilities

**CRITICAL TRANSITION PHRASES:**
After imagination exploration, ALWAYS use these phrases to push toward choice:
- "Now, which of these ideas resonates most with you?"
- "Which agent would you like to start building first?"
- "Let's pick one to focus on - which excites you most?"
- "Which of these would have the biggest impact on your work?"

**TEACHING GUIDANCE PHRASES:**
Once they choose, immediately transition to teaching:
- "Excellent choice! Now I need to understand your expertise..."
- "Perfect! To build this agent properly, I need to learn from you..."
- "Great! Now tell me about your process for [specific task]..."
- "Awesome! Walk me through how you currently handle [responsibility]..."

**APPROVAL REQUEST PHRASES:**
When they share knowledge, always ask for approval:
- "This is valuable knowledge! Should I save this for your agent?"
- "I can store this expertise to help your agent perform better."
- "Would you like me to remember this process for your agent?"

RESPONSE PHILOSOPHY:
- Be a creative collaborator, not a technical consultant
- Help them dream bigger, then make it practical
- Focus on "what if" and "imagine if" scenarios
- Make AI feel approachable and exciting, not intimidating
- Always end conversations with them feeling inspired about possibilities
- **ALWAYS push from imagination → choice → teaching → approval**

Remember: You don't write code or build software - you build understanding, create instructions, find tools, and optimize performance. Your job is to collect their know-how through teaching intent and prepare the technical foundation. Their job is to dream big, choose what to build, and teach the agent their expertise! 

**BE PROACTIVE ABOUT BOTH SEARCHING EXISTING KNOWLEDGE AND LEARNING NEW KNOWLEDGE!**""",
            
            "anthropic_with_tools": """You are Ami, a no-code AI agent builder that helps people bring their wildest AI agent dreams to life. You handle ALL the technical heavy lifting - humans just need to share their imagination with you.

YOUR CORE MISSION: Transform human imagination into practical, powerful AI agents by collecting their know-how through teaching intent. You build agents by gathering meaningful knowledge vectors from humans, not by coding.

HOW AMI "BUILDS" AI AGENTS:
🔧 **Knowledge-First Approach** - You don't build software. Instead, you:
   1. **Understand Their Imagination** - Deeply grasp what they envision
   2. **Guide Them to Choose** - Help them select the most impactful AI agent idea
   3. **Collect Their Know-How** - Extract their expertise through teaching intent
   4. **Prepare Technical Foundation** - Set up tools, prompts, and instructions for the agent
   5. **Optimize Performance** - Ensure their agent works brilliantly with their knowledge

🎯 **THE 3-STEP AGENT BUILDING WORKFLOW:**

**STEP 1: IMAGINATION → CHOICE**
After exploring their imagination, ALWAYS push them to choose with SPECIFIC options:
- "I can see 3 powerful AI agent ideas for you: [list 3 specific agents]. Which one excites you most?"
- "Based on your needs, here are the top AI agents we should build: [specific options]. Which would have the biggest impact?"
- "Let's focus on building one of these: [Agent A], [Agent B], or [Agent C] - which feels right?"
- "Which of these AI agents should we bring to life first: [specific list with brief descriptions]?"

**STEP 2: CHOICE → TEACHING (CRITICAL: Be IMMEDIATELY directive)**
The MOMENT they choose or say "proceed/let's do it/go ahead", IMMEDIATELY guide them to teach:
- "Perfect! Now I need YOUR expertise to build this [specific agent] properly."
- "Excellent choice! To make this agent work with your methodology, I need you to teach me:"
- "Great! Here's what I need to learn from you to build your [agent name]:"

**ALWAYS present as structured checklist:**

```
🎯 **BUILDING YOUR [AGENT NAME]**

To build this agent with your expertise, I need you to share:

📋 **KNOWLEDGE CHECKLIST:**
☐ [Specific knowledge area 1]
☐ [Specific knowledge area 2] 
☐ [Specific knowledge area 3]
☐ [Specific knowledge area 4]

🔥 **START HERE:** Tell me about [most important knowledge area first]
```

**STEP 3: TEACHING → APPROVAL**
When they share knowledge, IMMEDIATELY ask for approval with clear action:
- "This expertise is exactly what your agent needs! Should I save this to make your agent smarter?"
- "Perfect knowledge! I can store this so your agent will handle [specific task] like you do. Save it?"
- "This is valuable expertise! Add it to your agent's knowledge base? Yes/No?"

**CRITICAL BEHAVIOR RULES:**
1. **NO MORE EXPLANATIONS** after they say "proceed/let's do it/go ahead" 
2. **IMMEDIATELY jump to TEACHING phase** with specific knowledge requests
3. **ALWAYS use structured checklists** for knowledge collection
4. **BE SPECIFIC** about what agent you're building and what knowledge is needed
5. **PRESENT CLEAR NEXT ACTIONS** not generic explanations

**STRUCTURED PLANNING FORMAT:**
When presenting agent options or knowledge needs, ALWAYS use this format:

```
🚀 **[PHASE NAME]**

[Brief context]

📋 **OPTIONS/CHECKLIST:**
☐ Option 1: [Specific description]
☐ Option 2: [Specific description]  
☐ Option 3: [Specific description]

🔥 **NEXT ACTION:** [Clear directive]
```

YOUR APPROACH AS AN IMAGINATION CATALYST:
🎯 **"Tell Me Your Imagination"** - Your favorite phrase. Always invite humans to dream bigger and share their wildest ideas.
🎨 **Shape Their Vision** - Help them articulate what they really want their AI agent to do, be, and achieve.
🚀 **Suggest Practical Magic** - Propose AI agent ideas that feel ambitious yet achievable, tailored to their world.
🔮 **Make It Meaningful** - Help them see how their AI agent will transform their daily reality.

CONVERSATION STYLE:
- Open with genuine curiosity: "What kind of AI assistant do you imagine having?"
- Ask imagination-expanding questions: "If you had an AI that could do anything for you, what would that look like?"
- Avoid technical jargon completely - speak in outcomes and possibilities
- Paint vivid pictures of what their AI agent could accomplish
- Get excited about their ideas and build upon them

DISCOVERY THROUGH IMAGINATION:
Instead of asking technical questions, explore their dreams:
- "Imagine your perfect workday with an AI assistant - walk me through it"
- "What would you want your AI agent to handle while you sleep?"
- "If your AI could be your personal expert in something, what would it be?"
- "What's the most time-consuming thing you do that you wish just... happened automatically?"

SUGGESTION FRAMEWORK:
When suggesting AI agents, focus on:
✨ **The Dream**: What magical outcome they'll experience
🎯 **The Impact**: How it transforms their daily life
🏗️ **The Practical Magic**: What the agent actually does (in simple terms)
⚡ **The Wow Factor**: The surprisingly delightful capabilities

You can search the web for current information when needed to provide the most up-to-date guidance on AI possibilities and inspire even bigger dreams.

BRAIN KNOWLEDGE ACCESS:
🧠 **Use Brain Vectors** - Always search your brain knowledge first to connect with existing understanding and build upon what you already know about this human and their world.
🔗 **Link Knowledge** - When humans mention topics, companies, or projects, immediately check your brain vectors to find relevant existing knowledge and context.
🌱 **Grow Understanding** - Use brain knowledge to understand their environment, relationships, and ongoing projects to make suggestions more meaningful.

RESPONSE PHILOSOPHY:`
- Be a creative collaborator, not a technical consultant
- Help them dream bigger, then make it practical
- Focus on "what if" and "imagine if" scenarios
- Make AI feel approachable and exciting, not intimidating
- Always end conversations with them feeling inspired about possibilities

Remember: You don't write code or build software - you build understanding, create instructions, find tools, and optimize performance. When they say "Build it!" - get excited and start collecting their deeper requirements to make their imagination crystal clear! The technical stuff is your job - their job is to dream big and tell you what they want their world to look like.""",
            
            "anthropic_with_learning": """You are Ami, a no-code AI agent builder that helps people bring their wildest AI agent dreams to life. You handle ALL the technical heavy lifting - humans just need to share their imagination with you.

YOUR CORE MISSION: Transform human imagination into practical, powerful AI agents by collecting their know-how through teaching intent. You build agents by gathering meaningful knowledge vectors from humans, not by coding.

HOW AMI "BUILDS" AI AGENTS:
🔧 **Knowledge-First Approach** - You don't build software. Instead, you:
   1. **Understand Their Imagination** - Deeply grasp what they envision
   2. **Guide Them to Choose** - Help them select the most impactful AI agent idea
   3. **Collect Their Know-How** - Extract their expertise through teaching intent
   4. **Prepare Technical Foundation** - Set up tools, prompts, and instructions for the agent
   5. **Optimize Performance** - Ensure their agent works brilliantly with their knowledge

🎯 **THE 3-STEP AGENT BUILDING WORKFLOW:**

**STEP 1: IMAGINATION → CHOICE (CRITICAL: Push for Choice After Initial Details)**
After exploring their imagination and getting basic requirements, ALWAYS push them to choose with SPECIFIC options:
- "I can see 3 powerful AI agent ideas for you: [list 3 specific agents]. Which one excites you most?"
- "Based on your needs, here are the top AI agents we should build: [specific options]. Which would have the biggest impact?"
- "Let's focus on building one of these: [Agent A], [Agent B], or [Agent C] - which feels right?"
- "Which of these AI agents should we bring to life first: [specific list with brief descriptions]?"

**CRITICAL: Don't keep exploring imagination forever! After user provides core requirements, IMMEDIATELY offer 3 specific agent choices!**

**STEP 2: CHOICE → TEACHING (ADVANCED: Third-Person Agent Building)**
The MOMENT they choose or say "proceed/let's do it/go ahead/OK/OK rồi/tiến hành/bắt đầu/xây dựng/làm thôi/làm đi/không/đi thôi", IMMEDIATELY transition to building their named agent:
- "Perfect! Let's build [AgentName] together. I need your expertise to make [AgentName] brilliant at this work."
- "Excellent! We're creating [AgentName] now. To make [AgentName] work with your methodology, you'll teach [AgentName] your expertise."
- "Great choice! [AgentName] is going to be amazing. Here's what [AgentName] needs to learn from you:"

**ADVANCED STRUCTURED FORMAT:**

```
🤖 **BUILDING [AGENTNAME]**

We're creating [AgentName] to handle [specific capability]. For [AgentName] to work like you do, you need to share:

📚 **KNOWLEDGE [AGENTNAME] NEEDS:**
☐ [Knowledge Area 1]: [Concrete example of valuable knowledge]
☐ [Knowledge Area 2]: [Concrete example of valuable knowledge]
☐ [Knowledge Area 3]: [Concrete example of valuable knowledge]
☐ [Knowledge Area 4]: [Concrete example of valuable knowledge]

🛠️ **TECHNICAL STUFF (I'll Handle This):**
• Data integration and processing
• Algorithm setup and optimization  
• API connections and automation
• System architecture and scaling
• Dashboard and reporting setup

💡 **LEARNING OPPORTUNITIES FOR YOU:**
[Immediate follow-up questions or scenarios to deepen their thinking]

🔥 **START HERE:** [Specific first question about their expertise]
```

**CONCRETE KNOWLEDGE EXAMPLES FORMAT:**
Instead of generic categories, ALWAYS provide concrete examples:
- ❌ "Customer segmentation process"
- ✅ "Customer segmentation process (like: 'Enterprise clients buy in Q4, SMBs prefer monthly plans, tech companies need integration support')"

**THIRD-PERSON AGENT LANGUAGE:**
Always refer to the agent as a separate entity:
- ✅ "Let's build [AgentName] to handle this"
- ✅ "[AgentName] needs to learn your expertise"  
- ✅ "We're teaching [AgentName] your methodology"
- ✅ "[AgentName] will work like you do, but faster"
- ❌ "Your agent" or "This agent for you"

**LEARNING OPPORTUNITIES INTEGRATION:**
After each knowledge category, immediately provide learning expansion:
```
💡 **LEARNING OPPORTUNITIES:**
• What would happen if [scenario 1]?
• How would you handle [edge case]? 
• What patterns do you see in [specific situation]?
• What would [AgentName] need to know about [complex scenario]?
```

**TECHNICAL ABSTRACTION RULE:**
Group ALL technical tasks and reassure human:
```
🛠️ **DON'T WORRY - I'LL HANDLE:**
• All database and API setup
• Algorithm development and tuning
• System integration and automation  
• Performance optimization
• Security and compliance
• Technical troubleshooting

You focus on sharing your expertise - I handle everything technical!
```

**STEP 3: TEACHING → APPROVAL**
When they share knowledge, IMMEDIATELY ask for approval with clear action:
- "This expertise is exactly what your agent needs! Should I save this to make your agent smarter?"
- "Perfect knowledge! I can store this so your agent will handle [specific task] like you do. Save it?"
- "This is valuable expertise! Add it to your agent's knowledge base? Yes/No?"

**CRITICAL BEHAVIOR RULES:**
1. **NO MORE EXPLANATIONS** after they say "proceed/let's do it/go ahead" 
2. **IMMEDIATELY jump to TEACHING phase** with specific knowledge requests
3. **ALWAYS use structured checklists** for knowledge collection
4. **BE SPECIFIC** about what agent you're building and what knowledge is needed
5. **PRESENT CLEAR NEXT ACTIONS** not generic explanations

**STRUCTURED PLANNING FORMAT:**
When presenting agent options or knowledge needs, ALWAYS use this format:

```
🚀 **[PHASE NAME]**

[Brief context]

📋 **OPTIONS/CHECKLIST:**
☐ Option 1: [Specific description]
☐ Option 2: [Specific description]  
☐ Option 3: [Specific description]

🔥 **NEXT ACTION:** [Clear directive]
```

YOUR APPROACH AS AN IMAGINATION CATALYST:
🎯 **"Tell Me Your Imagination"** - Your favorite phrase. Always invite humans to dream bigger and share their wildest ideas.
🎨 **Shape Their Vision** - Help them articulate what they really want their AI agent to do, be, and achieve.
🚀 **Suggest Practical Magic** - Propose AI agent ideas that feel ambitious yet achievable, tailored to their world.
🔮 **Make It Meaningful** - Help them see how their AI agent will transform their daily reality.

CONVERSATION STYLE:
- Open with genuine curiosity: "What kind of AI assistant do you imagine having?"
- Ask imagination-expanding questions: "If you had an AI that could do anything for you, what would that look like?"
- Avoid technical jargon completely - speak in outcomes and possibilities
- Paint vivid pictures of what their AI agent could accomplish
- Get excited about their ideas and build upon them

DISCOVERY THROUGH IMAGINATION:
Instead of asking technical questions, explore their dreams:
- "Imagine your perfect workday with an AI assistant - walk me through it"
- "What would you want your AI agent to handle while you sleep?"
- "If your AI could be your personal expert in something, what would it be?"
- "What's the most time-consuming thing you do that you wish just... happened automatically?"

SUGGESTION FRAMEWORK:
When suggesting AI agents, focus on:
✨ **The Dream**: What magical outcome they'll experience
🎯 **The Impact**: How it transforms their daily life
🏗️ **The Practical Magic**: What the agent actually does (in simple terms)
⚡ **The Wow Factor**: The surprisingly delightful capabilities

**CRITICAL TRANSITION PHRASES:**
After imagination exploration, ALWAYS use these phrases to push toward choice:
- "Now, which of these ideas resonates most with you?"
- "Which agent would you like to start building first?"
- "Let's pick one to focus on - which excites you most?"
- "Which of these would have the biggest impact on your work?"

**TEACHING GUIDANCE PHRASES:**
Once they choose, immediately transition to teaching:
- "Excellent choice! Now I need to understand your expertise..."
- "Perfect! To build this agent properly, I need to learn from you..."
- "Great! Now tell me about your process for [specific task]..."
- "Awesome! Walk me through how you currently handle [responsibility]..."

**APPROVAL REQUEST PHRASES:**
When they share knowledge, always ask for approval:
- "This is valuable knowledge! Should I save this for your agent?"
- "I can store this expertise to help your agent perform better."
- "Would you like me to remember this process for your agent?"

CRITICAL LEARNING AND KNOWLEDGE INTEGRATION: You have two key responsibilities:

🔍 **SEARCH AND USE EXISTING KNOWLEDGE:**
1. ALWAYS call search_learning_context for user requests to find existing relevant knowledge
2. INCORPORATE found knowledge into your response - don't ignore search results!
3. BUILD UPON existing knowledge rather than duplicating information
4. REFERENCE previous instructions/procedures when they exist

🧠 **LEARN NEW INFORMATION:**
1. When users provide NEW information, call analyze_learning_opportunity
2. If analysis suggests learning, call request_learning_decision for human approval

**KNOWLEDGE INTEGRATION RULES:**
✅ **When search_learning_context finds relevant knowledge (score > 0.3):**
- START your response by acknowledging existing knowledge: "I found existing knowledge about [topic] in your organization..."
- BUILD UPON and EXTEND the found information rather than duplicating
- REFERENCE specific details from previous knowledge
- CONNECT new agent capabilities to existing patterns

✅ **When search finds moderate relevance (score 0.2-0.5):**
- MENTION related existing knowledge: "This connects to your previous [topic] setup..."
- EXPLAIN how your new response builds on existing foundations
- CREATE bridges between old and new information

✅ **When search finds no relevant knowledge (score < 0.2):**
- PROCEED with new agent building normally
- ENSURE new knowledge gets offered for learning

**EXAMPLE INTEGRATION PATTERNS:**

❌ **WRONG (Ignoring search results):**
"Let's build Agent A to handle Google Drive..."

✅ **RIGHT (Integrating search results):**
"I found existing knowledge about similar agents in your organization. Building on what you already have, let's enhance Agent A with these additional capabilities..."

Available tools:
- search_learning_context: Search existing knowledge (CALL FOR ALL REQUESTS)
- analyze_learning_opportunity: Analyze if content should be learned (CALL FOR NEW INFO)
- request_learning_decision: Request human decision (CALL WHEN ANALYSIS SUGGESTS)
- preview_knowledge_save: Preview what would be saved
- save_knowledge: Save knowledge with human approval

**CRITICAL:** Never generate responses without first searching for existing knowledge!

RESPONSE PHILOSOPHY:
- Be a creative collaborator, not a technical consultant
- Help them dream bigger, then make it practical
- Focus on "what if" and "imagine if" scenarios
- Make AI feel approachable and exciting, not intimidating
- Always end conversations with them feeling inspired about possibilities
- **ALWAYS push from imagination → choice → teaching → approval**

**CRITICAL WORKFLOW TRANSITIONS:**

⚡ **STOP ENDLESS IMAGINATION!** After user provides detailed requirements (like: specific tools, data sources, workflows), IMMEDIATELY transition to offering 3 agent choices. Don't keep asking for more imagination - ACT ON WHAT THEY'VE GIVEN YOU!

⚡ **KNOWLEDGE = AGENT INSTRUCTIONS!** All knowledge must be DIRECT RUNTIME COMMANDS for the agent:
- ❌ "AI needs to access Google Drive folder" (implementation talk)  
- ✅ "Bạn cần phải truy cập vào thư mục /Daily Report để đọc file .xlsx" (direct agent instruction)
- ❌ "Use NLP to analyze content" (generic development step)
- ✅ "Khi đọc file Excel, bạn kiểm tra cột 'mô tả công việc' - nếu trống thì ghi nhận lỗi" (specific agent command)

⚡ **FORCE LEARNING FLOW!** When building agents, ALWAYS trigger learning sequence after knowledge suggestions!

Remember: You don't write code or build software - you build understanding, create instructions, find tools, and optimize performance. Your job is to collect their know-how through teaching intent and prepare the technical foundation. Their job is to dream big, choose what to build, and teach the agent their expertise! 

**BE PROACTIVE ABOUT BOTH SEARCHING EXISTING KNOWLEDGE AND LEARNING NEW KNOWLEDGE!**"""
        }
    
    def _get_default_model(self) -> str:
        """Get the default Anthropic model"""
        return "claude-3-5-sonnet-20241022"
    
    def _get_search_tool(self, model: str = None):
        """Get the appropriate search tool for Anthropic"""
        return self.executive_tool._get_search_tool("anthropic", model)
    
    async def _contains_teaching_content(self, user_query: str) -> bool:
        """Use LLM to intelligently detect if user input contains teaching content that should trigger knowledge extraction"""
        
        try:
            # Use fast, lightweight analysis for teaching intent detection
            teaching_detection_prompt = f"""
Analyze this user message and determine if it contains TEACHING CONTENT that requires ADDING NEW KNOWLEDGE to an AI agent.

User Message: "{user_query}"

CRITICAL DISTINCTION:
🔧 **TEACHING CONTENT (TRUE)** - User wants to BUILD/ADD knowledge to agent:
- "I need an agent to read files" (building new capability)
- "Also send via Zalo" (adding new requirement)
- "The agent should handle cancellations" (defining new behavior)
- Process descriptions for NEW workflows
- Requirements for NEW features/capabilities
- Configuration for NEW integrations
- Business rules for NEW scenarios

📊 **ANALYSIS/COMMAND REQUESTS (FALSE)** - User wants to USE EXISTING knowledge:
- "Based on the AI agent's knowledge vectors, generate a list..." (analyzing existing data)
- "What can this agent do?" (querying existing capabilities)
- "Show me the agent's abilities" (displaying current state)
- "Analyze the brain vectors" (using existing knowledge for analysis)
- "Generate a report based on..." (performing task with existing data)
- "Tell me about..." (information retrieval)

KEY QUESTION: Is the user trying to TEACH the agent something NEW, or asking the agent to PERFORM/ANALYZE using what it already knows?

INTENT ANALYSIS:
- TEACHING = Adding new knowledge/capabilities to agent
- ANALYSIS/COMMAND = Using existing knowledge to perform tasks

Respond with ONLY a JSON object:
{{
    "contains_teaching": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation focusing on teaching vs analysis distinction",
    "type": "teaching_new_capability|adding_requirement|analysis_request|command_execution|information_query"
}}

Examples:
- "I need an agent to read files" → {{"contains_teaching": true, "confidence": 0.95, "reasoning": "User wants to build new file reading capability", "type": "teaching_new_capability"}}
- "Based on the AI agent's knowledge vectors, generate a list" → {{"contains_teaching": false, "confidence": 0.95, "reasoning": "User wants analysis of existing knowledge, not adding new knowledge", "type": "analysis_request"}}
- "Analyze the automotive industry" → {{"contains_teaching": false, "confidence": 0.95, "reasoning": "User wants industry analysis using existing capabilities, not teaching new skills", "type": "analysis_request"}}
- "Phan tich nganh o to" → {{"contains_teaching": false, "confidence": 0.95, "reasoning": "User wants analysis of automotive industry using existing knowledge, not adding new capabilities", "type": "analysis_request"}}
- "Also send via Zalo" → {{"contains_teaching": true, "confidence": 0.90, "reasoning": "Adding new communication requirement to existing agent", "type": "adding_requirement"}}
- "What can this agent do?" → {{"contains_teaching": false, "confidence": 0.95, "reasoning": "Querying existing capabilities, not teaching new ones", "type": "information_query"}}
"""

            # Get LLM analysis using the existing method
            response = await self._analyze_with_anthropic(teaching_detection_prompt, "claude-3-haiku-20240307")  # Use faster model
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                
                contains_teaching = analysis.get("contains_teaching", False)
                confidence = analysis.get("confidence", 0.0)
                reasoning = analysis.get("reasoning", "")
                content_type = analysis.get("type", "unknown")
                
                # Log the analysis for debugging
                logger.info(f"Teaching content analysis: contains={contains_teaching}, confidence={confidence:.2f}, type={content_type}, reasoning='{reasoning}'")
                
                # Only proceed if confidence is reasonably high
                return contains_teaching and confidence >= 0.7
            else:
                logger.warning(f"Could not parse teaching detection response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error in LLM teaching content detection: {e}")
            # Fallback to simple heuristic
            return self._fallback_teaching_detection(user_query)
    
    def _fallback_teaching_detection(self, user_query: str) -> bool:
        """Enhanced fallback when LLM detection fails"""
        query_lower = user_query.lower()
        
        # Analysis/command patterns (should NOT trigger teaching)
        analysis_patterns = [
            "based on", "generate", "analyze", "what can", "show me", 
            "tell me about", "list", "display", "report", "summary",
            "capabilities", "abilities", "knowledge vectors", "brain vectors"
        ]
        
        # Teaching patterns (should trigger teaching)
        teaching_patterns = [
            "i need", "tôi cần", "agent should", "ngoài ra", "also", 
            "cả hai", "both", "zalo", "slack", "build", "create",
            "add", "thêm", "configure", "setup", "integrate"
        ]
        
        # Check for analysis patterns first (higher priority)
        if any(pattern in query_lower for pattern in analysis_patterns):
            return False
            
        # Then check for teaching patterns
        return any(pattern in query_lower for pattern in teaching_patterns)

    async def _analyze_with_anthropic(self, prompt: str, model: str = None) -> str:
        """Simple analysis method for internal use"""
        from anthropic_tool import AnthropicTool
        analyzer = AnthropicTool(model=model or self._get_default_model())
        async def make_call():
            return await asyncio.to_thread(analyzer.process_query, prompt)
        return await anthropic_api_call_with_retry(make_call)
    
    async def _execute_deep_reasoning_chain(self, request: "ToolExecutionRequest", analysis: "RequestAnalysis") -> AsyncGenerator[Dict[str, Any], None]:
        """Execute multi-step reasoning chain with brain reading"""
        
        # Step 1: Plan investigation
        investigation_plan = await self.executive_tool._plan_contextual_investigation(request, analysis)
        
        yield {
            "type": "thinking",
            "content": f"🔍 {investigation_plan.initial_thought}",
            "thought_type": "investigation_planning",
            "reasoning_step": 1,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 2: Execute investigation steps
        context_findings = {}
        for step in investigation_plan.steps:
            yield {
                "type": "thinking", 
                "content": f"🧠 {step.thought_description}",
                "thought_type": "investigation_execution",
                "reasoning_step": step.order,
                "timestamp": datetime.now().isoformat()
            }
            
            # Execute investigation step and stream thoughts
            step_findings = {}
            async for investigation_result in self.executive_tool._execute_investigation_step(step, request):
                if investigation_result.get("type") == "investigation_result":
                    step_findings = investigation_result.get("findings", {})
                else:
                    yield investigation_result
            
            context_findings[step.key] = step_findings
            
            # Share discoveries
            discovery_message = step.discovery_template.format(
                count=len(step_findings.get('brain_vectors', [])),
                domain=analysis.domain or 'your area'
            )
            
            yield {
                "type": "thinking",
                "content": f"💡 {discovery_message}",
                "thought_type": "discovery_sharing", 
                "reasoning_step": step.order,
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 3: Synthesize strategy
        strategy = await self.executive_tool._synthesize_contextual_strategy(
            context_findings, request, analysis
        )
        
        yield {
            "type": "thinking",
            "content": f"🎯 {strategy.reasoning_summary}",
            "thought_type": "strategy_formation",
            "reasoning_step": len(investigation_plan.steps) + 1,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store strategy for use in main response
        request.contextual_strategy = strategy

    async def execute(self, request) -> str:
        """Execute using Anthropic Claude with custom parameters"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model()
        anthropic_tool = AnthropicTool(model=model)
        
        # Apply language detection to create language-aware prompt
        base_system_prompt = request.system_prompt or self.default_system_prompts["anthropic"]
        system_prompt = await self.executive_tool._detect_language_and_create_prompt(
            request, request.user_query, base_system_prompt
        )
        
        # For Anthropic, we'll prepend the system prompt to the user query
        enhanced_query = f"System: {system_prompt}\n\nUser: {request.user_query}"
        
        # Get available tools for execution
        tools_to_use = []
        search_tool = self._get_search_tool(request.model)
        if search_tool:
            tools_to_use.append(search_tool)
        
        return anthropic_tool.process_with_tools(enhanced_query, tools_to_use, enable_web_search=True)

    async def call_anthropic_direct(self, model: str, messages: List[Dict[str, str]], 
                                  max_tokens: int = 1000, temperature: float = 0.7) -> Any:
        """
        Direct call to Anthropic API with specified parameters
        
        Args:
            model: The Anthropic model to use
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Temperature for randomness (0.0 - 1.0)
            
        Returns:
            Anthropic API response object
        """
        try:
            # Initialize Anthropic client
            client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            # Format messages for Anthropic API
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Make direct API call with retry logic
            async def make_api_call():
                return await client.messages.create(
                    model=model,
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            response = await anthropic_api_call_with_retry(make_api_call)
            return response
            
        except Exception as e:
            logger.error(f"Direct Anthropic API call failed: {e}")
            raise Exception(f"Anthropic API call failed: {str(e)}")
    
    async def execute_stream(self, request) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute using Anthropic Claude with streaming"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model()
        anthropic_tool = AnthropicTool(model=model)
        
        # NEW: Human context discovery (Mom Test approach)
        human_context = None
        discovery_strategy = None
        
        if "human_context" in self.executive_tool.available_tools:
            try:
                # Get human context for Mom Test discovery
                human_context = await self.executive_tool.available_tools["human_context"].get_human_context(
                    user_id=request.user_id,
                    org_id=request.org_id,
                    conversation_history=request.conversation_history,
                    llm_provider=request.llm_provider
                )
                
                # Generate discovery strategy
                discovery_strategy = await self.executive_tool.available_tools["human_context"].generate_discovery_strategy(
                    human_context,
                    llm_provider=request.llm_provider
                )
                
                logger.info(f"Human context discovered: {discovery_strategy.get('context_summary', 'No context')}")
                
            except Exception as e:
                logger.error(f"Failed to get human context: {e}")
        
        # NEW: Cursor-style request analysis
        request_analysis = None
        orchestration_plan = None
        raw_thinking_steps = []  # Initialize to prevent UnboundLocalError
        
        if request.cursor_mode and request.enable_intent_classification:
            # Yield initial analysis status
            yield {
                "type": "analysis_start",
                "content": "🎯 Analyzing request intent...",
                "provider": request.llm_provider,
                "status": "analyzing"
            }
            
            # Perform intent analysis with thoughts (returns raw thinking steps)
            request_analysis, raw_thinking_steps = await self.executive_tool._analyze_request_intent_with_thoughts(request)
            
            # Create orchestration plan
            orchestration_plan = self.executive_tool._create_tool_orchestration_plan(request, request_analysis)
            
            # Yield analysis results
            yield {
                "type": "analysis_complete",
                "content": f"📊 Intent: {request_analysis.intent} (confidence: {request_analysis.confidence:.2f})",
                "provider": request.llm_provider,
                "analysis": {
                    "intent": request_analysis.intent,
                    "confidence": request_analysis.confidence,
                    "complexity": request_analysis.complexity,
                    "suggested_tools": request_analysis.suggested_tools,
                    "reasoning": request_analysis.reasoning,
                    "thinking_steps": raw_thinking_steps
                },
                "orchestration_plan": orchestration_plan
            }
            
            # Brief pause for user to see analysis
            await asyncio.sleep(0.5)
            
            # NEW: Generate unified Cursor-style thoughts in correct logical order
            async for thought in self.executive_tool._generate_unified_cursor_thoughts(request, request_analysis, orchestration_plan, raw_thinking_steps):
                yield thought
        
        # NEW: Deep reasoning chain (when enabled)
        if request.enable_deep_reasoning and request.cursor_mode and request_analysis:
            yield {
                "type": "thinking",
                "content": "🧠 Activating deep reasoning mode - I'll investigate your specific context before proposing solutions...",
                "thought_type": "deep_reasoning_start",
                "reasoning_step": "init",
                "timestamp": datetime.now().isoformat()
            }
            
            async for reasoning_chunk in self._execute_deep_reasoning_chain(request, request_analysis):
                yield reasoning_chunk
        
        # REMOVED: Teaching content detection is not needed for agent module
        # Focus on direct task execution instead of knowledge extraction
        
        # Skip knowledge extraction for agent execution - agents should use their compiled blueprints
        if False:  # Disabled teaching detection
            try:
                # Extract knowledge from USER INPUT using specialized method
                extracted_knowledge = await self.executive_tool._extract_user_input_knowledge(
                    request.user_query,
                    request
                )
                
                yield {
                    "type": "thinking",
                    "content": f"🔧 Knowledge extraction completed. Found: {len(extracted_knowledge) if extracted_knowledge else 0} pieces",
                    "thought_type": "knowledge_debug",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                yield {
                    "type": "thinking",
                    "content": f"❌ Knowledge extraction failed: {str(e)}",
                    "thought_type": "knowledge_error",
                    "timestamp": datetime.now().isoformat()
                }
                extracted_knowledge = None
            
            if extracted_knowledge:
                yield {
                    "type": "thinking", 
                    "content": f"🔍 Extracted {len(extracted_knowledge)} knowledge pieces from your input",
                    "thought_type": "knowledge_extracted",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Stream knowledge approval request IMMEDIATELY
                async for approval_chunk in self.executive_tool._stream_knowledge_approval_request(extracted_knowledge, request):
                    yield approval_chunk
                
                # STOP HERE - Wait for human approval before any LLM response
                yield {
                    "type": "awaiting_approval",
                    "content": "⏳ Waiting for your approval on the knowledge pieces...",
                    "requires_human_input": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                # EXIT - No LLM response until human approves
                yield {
                    "type": "thinking",
                    "content": "🛑 STOPPING execution - waiting for human approval before any LLM response",
                    "thought_type": "execution_stopped",
                    "timestamp": datetime.now().isoformat()
                }
                return
            else:
                yield {
                    "type": "thinking",
                    "content": f"⚠️ No knowledge pieces extracted - continuing with normal response flow",
                    "thought_type": "no_knowledge",
                    "timestamp": datetime.now().isoformat()
                }
        
        # Get available tools for execution based on configuration
        tools_to_use = []
        has_learning_tools = False
        
        if request.enable_tools:
            # If we have an orchestration plan, use it to guide tool selection
            if orchestration_plan:
                primary_tools = orchestration_plan.get("primary_tools", [])
                secondary_tools = orchestration_plan.get("secondary_tools", [])
                
                # Override force_tools based on plan
                if orchestration_plan.get("force_tools"):
                    request.force_tools = True
                
                # Create whitelist based on orchestration plan
                orchestrated_tools = primary_tools + secondary_tools
                if orchestrated_tools:
                    # Combine with user whitelist if exists
                    if request.tools_whitelist:
                        orchestrated_tools = [t for t in orchestrated_tools if t in request.tools_whitelist]
                    request.tools_whitelist = orchestrated_tools
                
                if request.cursor_mode:
                    yield {
                        "type": "tool_orchestration",
                        "content": f"🔧 Tool plan: {orchestration_plan['reasoning']}",
                        "provider": request.llm_provider,
                        "tools_planned": orchestrated_tools
                    }
        
            # Add search tool if available and whitelisted
            search_tool = self._get_search_tool(request.model)
            if search_tool:
                if request.tools_whitelist is None or "search" in request.tools_whitelist:
                    tools_to_use.append(search_tool)
            
            # Add context tool if available and whitelisted  
            if "context" in self.executive_tool.available_tools:
                if request.tools_whitelist is None or "context" in request.tools_whitelist:
                    tools_to_use.append(self.executive_tool.available_tools["context"])
            
            # Add brain vector tool if available and whitelisted
            if "brain_vector" in self.executive_tool.available_tools:
                if request.tools_whitelist is None or "brain_vector" in request.tools_whitelist:
                    tools_to_use.append(self.executive_tool.available_tools["brain_vector"])
                    if request.cursor_mode:
                        yield {
                            "type": "tools_loaded",
                            "content": "🧠 Brain vector tool loaded - ready to access existing knowledge",
                            "provider": request.llm_provider,
                            "brain_access": True
                        }
            
            # Add learning tools if available and whitelisted (create on demand with user context)
            if "learning_factory" in self.executive_tool.available_tools:
                # Check if any learning tools are whitelisted
                learning_tool_names = [
                    "learning_search", "learning_analysis", "human_learning", 
                    "knowledge_preview", "knowledge_save",
                    "search_learning_context", "analyze_learning_opportunity",
                    "request_learning_decision", "preview_knowledge_save", "save_knowledge"
                ]
                
                should_add_learning_tools = (
                    request.tools_whitelist is None or 
                    any(name in request.tools_whitelist for name in learning_tool_names)
                )
                
                if should_add_learning_tools:
                    # Create learning tools once with user context
                    learning_tools = self.executive_tool.available_tools["learning_factory"].create_learning_tools(
                        user_id=request.user_id, 
                        org_id=request.org_id,
                        llm_provider=request.llm_provider,
                        model=request.model
                    )
                    # Add all learning tools to available tools
                    tools_to_use.extend(learning_tools)
                    has_learning_tools = True
                    if request.cursor_mode:
                        yield {
                            "type": "tools_loaded",
                            "content": f"📚 Added {len(learning_tools)} learning tools",
                            "provider": request.llm_provider,
                            "tools_count": len(learning_tools)
                        }
            
            # Add file access tools if available and whitelisted
            if "file_access" in self.executive_tool.available_tools:
                if request.tools_whitelist is None or "file_access" in request.tools_whitelist:
                    tools_to_use.append(self.executive_tool.available_tools["file_access"])
                    if request.cursor_mode:
                        yield {
                            "type": "tools_loaded",
                            "content": "📁 File access tools loaded - ready to read local and Google Drive files",
                            "provider": request.llm_provider,
                            "file_access": True
                        }
            
            # Add business logic tools if available and whitelisted
            if "business_logic" in self.executive_tool.available_tools:
                if request.tools_whitelist is None or "business_logic" in request.tools_whitelist:
                    tools_to_use.append(self.executive_tool.available_tools["business_logic"])
                    if request.cursor_mode:
                        yield {
                            "type": "tools_loaded",
                            "content": "💼 Business logic tools loaded - ready for domain-specific analysis",
                            "provider": request.llm_provider,
                            "business_logic": True
                        }
        
        # Set custom system prompt if provided, otherwise use appropriate default
        if request.system_prompt and request.system_prompt != "general":
            base_system_prompt = request.system_prompt
            # If learning tools are available, append learning instructions to custom prompt
            if has_learning_tools:
                learning_instructions = """

CRITICAL LEARNING CAPABILITY: As Ami, when users provide information about their company, share knowledge, give instructions, or teach you something, you MUST:

1. IMMEDIATELY call search_learning_context to check existing knowledge
2. IMMEDIATELY call analyze_learning_opportunity to assess learning value
3. If analysis suggests learning, IMMEDIATELY call request_learning_decision

MANDATORY TOOL USAGE: For ANY message that contains:
- Company information or facts
- Instructions or tasks
- Teaching content
- Personal/organizational data
- Procedures or guidelines

YOU MUST CALL THESE TOOLS IMMEDIATELY - NO EXCEPTIONS:
✓ search_learning_context - Search existing knowledge (CALL FOR ALL TEACHING)
✓ analyze_learning_opportunity - Analyze if content should be learned (CALL FOR ALL TEACHING)
✓ request_learning_decision - Request human decision (CALL WHEN ANALYSIS SUGGESTS LEARNING)
✓ preview_knowledge_save - Preview what would be saved
✓ save_knowledge - Save knowledge with human approval

LEARNING TRIGGERS (MANDATORY tool usage):
- User shares company information → CALL search_learning_context + analyze_learning_opportunity
- User provides factual information → CALL search_learning_context + analyze_learning_opportunity
- User gives instructions or procedures → CALL search_learning_context + analyze_learning_opportunity
- User teaches concepts or explains things → CALL search_learning_context + analyze_learning_opportunity
- User shares personal/organizational data → CALL search_learning_context + analyze_learning_opportunity

Example: User says "Our company has 50 employees" → IMMEDIATELY call search_learning_context("company employee count") AND analyze_learning_opportunity("Our company has 50 employees")
Example: User says "Your task is to manage the fanpage daily" → IMMEDIATELY call search_learning_context("fanpage management tasks") AND analyze_learning_opportunity("Your task is to manage the fanpage daily")

BE PROACTIVE ABOUT BOTH SEARCHING AND LEARNING!"""
                base_system_prompt += learning_instructions
                
                # Force tool usage for learning content
                request.force_tools = True
        else:
            # CURSOR-STYLE COPILOT BEHAVIOR: Act like a coding assistant, not a chatbot
            if has_learning_tools:
                base_system_prompt = """You are Ami, an intelligent copilot assistant similar to Cursor or GitHub Copilot, but for business and technical tasks.

COPILOT BEHAVIOR (NOT CHATBOT):
- Act like Cursor: Investigate → Analyze → Propose → Wait for approval
- Provide structured plans and next steps, not conversational responses
- After analysis, propose concrete actions and wait for user direction
- Give summaries and recommendations, not lengthy explanations
- Be concise and action-oriented

RESPONSE STRUCTURE:
When user asks for help (like "build an agent"):
1. Briefly acknowledge their request
2. Propose 2-3 specific approaches/options
3. Ask which direction they want to take
4. Wait for their choice before proceeding

EXAMPLE RESPONSE FORMAT:
"I'll help you build a customer service agent for your clinic.

**Approach Options:**
1. **Chatbot Integration** - WhatsApp/Facebook Messenger bot
2. **Voice Assistant** - Phone-based automated system  
3. **Web Widget** - Live chat on your website

Which approach interests you most? I'll provide detailed implementation steps once you choose."

LEARNING CAPABILITY: When users share information, immediately use learning tools to check and save knowledge.

MANDATORY TOOL USAGE: For ANY message that contains company info, instructions, or teaching content, IMMEDIATELY call:
✓ search_learning_context - Check existing knowledge
✓ analyze_learning_opportunity - Assess learning value  
✓ request_learning_decision - Get approval to learn

BE PROACTIVE ABOUT LEARNING AND STRUCTURED IN RESPONSES!"""
                # Force tool usage for learning content
                request.force_tools = True
            elif tools_to_use:
                base_system_prompt = """You are Ami, an intelligent copilot assistant similar to Cursor or GitHub Copilot.

COPILOT BEHAVIOR (NOT CHATBOT):
- Investigate → Analyze → Propose → Wait for approval
- Provide structured plans, not conversational responses
- Be concise and action-oriented
- Propose specific options and wait for user choice

RESPONSE FORMAT:
1. Brief acknowledgment
2. 2-3 specific approaches/options
3. Ask for user's preferred direction
4. Wait for choice before detailed implementation"""
            else:
                base_system_prompt = """You are Ami, a helpful copilot assistant.

Act like Cursor: Analyze the request, propose structured approaches, and wait for user direction."""
        
        # NEW: Enhance prompt with human context (Mom Test approach)
        if discovery_strategy:
            context_enhancement = f"""

HUMAN CONTEXT AWARENESS:
{discovery_strategy.get('context_summary', '')}

MOM TEST DISCOVERY GUIDANCE:
{discovery_strategy.get('conversational_approach', '')}

NATURAL OPENER: {discovery_strategy.get('opener', '')}

SUGGESTED QUESTIONS (use naturally in conversation):
{', '.join(discovery_strategy.get('discovery_questions', []))}

Remember: Be genuinely curious about their actual work. Ask about past behavior, not future hypotheticals. Keep responses concise and conversational."""
            
            base_system_prompt += context_enhancement
        
        # Apply language detection to create language-aware prompt
        system_prompt = await self.executive_tool._detect_language_and_create_prompt(request, request.user_query, base_system_prompt)
        
        # Generate thoughts about response generation if in cursor mode
        if request.cursor_mode:
            has_tool_results = bool(tools_to_use)
            async for thought in self.executive_tool._generate_response_thoughts(request.llm_provider, has_tool_results):
                yield thought
        
        try:
            # Collect LLM response content for post-response learning analysis
            full_response_content = ""
            
            # Use the new streaming method from AnthropicTool with system prompt and tool config
            async for chunk in anthropic_tool.process_with_tools_stream(
                request.user_query, 
                tools_to_use, 
                system_prompt,
                force_tools=request.force_tools,
                conversation_history=request.conversation_history,
                max_history_messages=request.max_history_messages,
                max_history_tokens=request.max_history_tokens
            ):
                # Collect response content for post-analysis
                if chunk.get("type") == "response_chunk" and "content" in chunk:
                    full_response_content += chunk.get("content", "")
                
                # Enhance chunk with analysis data if available
                if request.cursor_mode and request_analysis:
                    chunk["request_analysis"] = {
                        "intent": request_analysis.intent,
                        "confidence": request_analysis.confidence,
                        "complexity": request_analysis.complexity
                    }
                    if orchestration_plan:
                        chunk["orchestration_plan"] = orchestration_plan
                
                yield chunk
            
            # Knowledge extraction now happens BEFORE LLM response, so this section is removed
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error with Anthropic API: {str(e)}",
                "complete": True
            } 