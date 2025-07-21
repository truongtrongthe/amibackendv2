"""
Interactive Learning Tools - Phase 1 Implementation

This module provides explicit learning tools that the LLM can call to make
the learning process interactive and human-guided, replacing the monolithic
hidden learning logic.
"""

import os
import json
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global dictionary to store pending learning decisions (in-memory for now)
PENDING_LEARNING_DECISIONS: Dict[str, Dict[str, Any]] = {}

def get_pending_knowledge_decisions(user_id: str = None) -> List[Dict[str, Any]]:
    """Get all pending learning decisions, optionally filtered by user_id"""
    decisions = []
    for decision_id, decision_data in PENDING_LEARNING_DECISIONS.items():
        if user_id is None or decision_data.get("user_id") == user_id:
            # Only return decisions that are still pending
            if decision_data.get("status") == "PENDING":
                decisions.append({
                    "decision_id": decision_id,
                    **decision_data
                })
    
    logger.info(f"Found {len(decisions)} pending decisions for user {user_id or 'all'}")
    return decisions

def clear_pending_knowledge_decisions():
    """Clear all pending decisions (for testing)"""
    global PENDING_LEARNING_DECISIONS
    PENDING_LEARNING_DECISIONS.clear()
    logger.info("Cleared all pending learning decisions")

@dataclass
class LearningContext:
    """Context for learning operations"""
    user_id: str
    org_id: str
    llm_provider: str = "openai"  # Default to OpenAI for GPT-4o
    model: str = "gpt-4o"  # Use GPT-4o for better quality

class LearningSearchTool:
    """Tool for LLM to search existing learning context"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown", llm_provider: str = "openai", model: str = "gpt-4o"):
        self.user_id = user_id
        self.org_id = org_id
        self.llm_provider = llm_provider
        self.model = model or "gpt-4o"  # Default to GPT-4o
        self.name = "learning_search"
    
    def search_learning_context(self, query: str, context: str = "", limit: int = 5) -> str:
        """
        LLM calls this to search for similar existing knowledge before learning something new
        
        Args:
            query: Search query to find similar knowledge
            context: Additional context for the search
            limit: Maximum number of results to return
            
        Returns:
            Formatted search results
        """
        try:
            logger.info(f"LearningSearchTool: Searching for '{query[:50]}...' with limit {limit}")
            
            # Import here to avoid circular imports
            from pccontroller import query_knowledge
            
            # Search for existing knowledge - handle async context properly
            try:
                import asyncio
                # Check if we're in an async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, but this is a sync method called by LLM tools
                    # Create a new task but don't use run_until_complete
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    exception_queue = queue.Queue()
                    
                    def run_async_query():
                        try:
                            # Create new event loop for this thread
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                results = new_loop.run_until_complete(query_knowledge(
                                    query=query,
                                    org_id=self.org_id,
                                    user_id=self.user_id,
                                    top_k=limit,
                                    min_similarity=0.3
                                ))
                                result_queue.put(results)
                            finally:
                                new_loop.close()
                        except Exception as e:
                            exception_queue.put(e)
                    
                    # Run in a separate thread
                    thread = threading.Thread(target=run_async_query)
                    thread.start()
                    thread.join(timeout=10)  # 10 second timeout
                    
                    if not exception_queue.empty():
                        raise exception_queue.get()
                    
                    if not result_queue.empty():
                        results = result_queue.get()
                    else:
                        raise Exception("Query timed out")
                        
                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    results = asyncio.run(query_knowledge(
                        query=query,
                        org_id=self.org_id,
                        user_id=self.user_id,
                        top_k=limit,
                        min_similarity=0.3
                    ))
                    
            except Exception as async_error:
                logger.error(f"Async query failed: {async_error}")
                # Fallback: return a message indicating search wasn't possible
                return f"SEARCH_RESULTS: Unable to search existing knowledge due to async context error. Treating as new information for query '{query}'."
            
            if not results:
                return f"SEARCH_RESULTS: No existing knowledge found for '{query}'. This appears to be new information."
            
            # Format results for LLM analysis
            formatted_results = []
            for i, result in enumerate(results, 1):
                similarity = result.get('score', 0.0)
                content = result.get('raw', '')[:200] + "..." if len(result.get('raw', '')) > 200 else result.get('raw', '')
                categories = result.get('categories', [])
                
                formatted_results.append(f"""
RESULT {i} (Similarity: {similarity:.3f}):
Content: {content}
Categories: {', '.join(categories)}
Created: {result.get('created_at', 'Unknown')}
""")
            
            results_text = "\n".join(formatted_results)
            
            # Calculate max similarity for decision making
            max_similarity = max([r.get('score', 0.0) for r in results]) if results else 0.0
            
            return f"""SEARCH_RESULTS for query "{query}":

Found {len(results)} similar knowledge entries (Max similarity: {max_similarity:.3f}):

{results_text}

ANALYSIS:
- High similarity (>0.7): Very similar content exists - consider updating instead of creating new
- Medium similarity (0.3-0.7): Some related content exists - new information might be valuable
- Low similarity (<0.3): No similar content found - new information is likely valuable
"""
            
        except Exception as e:
            logger.error(f"Learning search error: {e}")
            return f"SEARCH_ERROR: Failed to search existing knowledge - {str(e)}. Treating as new information."

class LearningAnalysisTool:
    """Tool for LLM to analyze if something should be learned"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown", llm_provider: str = "openai", model: str = "gpt-4o"):
        self.user_id = user_id
        self.org_id = org_id
        self.llm_provider = llm_provider
        self.model = model or "gpt-4o"  # Default to GPT-4o
        self.name = "learning_analysis"
    
    async def analyze_learning_opportunity(self, user_message: str, conversation_context: str = "", search_results: str = "") -> str:
        """
        LLM calls this to analyze if user input should be learned/saved
        
        Args:
            user_message: The user's message to analyze
            conversation_context: Previous conversation context
            search_results: Results from searching existing knowledge
            
        Returns:
            Analysis result with recommendation
        """
        
        try:
            logger.info(f"LearningAnalysisTool: Analyzing message '{user_message[:50]}...'")
            
            # Analyze different aspects
            teaching_intent = await self._detect_teaching_intent(user_message, conversation_context)
            knowledge_gap = self._detect_knowledge_gap(user_message, search_results)
            learning_value = self._assess_learning_value(user_message)
            content_quality = self._assess_content_quality(user_message)
            similarity_score = self._extract_similarity_from_search(search_results)
            
            # Generate overall recommendation
            recommendation = self._generate_learning_recommendation(
                teaching_intent, knowledge_gap, learning_value, content_quality, similarity_score
            )
            
            # Log detailed analysis results for debugging
            reasoning = teaching_intent.get('reasoning', 'no_reasoning_provided')
            logger.info(f"LLM Teaching Intent Analysis - Detected: {teaching_intent['detected']} ({teaching_intent['confidence']:.2f}), "
                       f"Indicators: {teaching_intent['indicators']}, Reasoning: {reasoning}, "
                       f"Final Action: {recommendation['action']}, Confidence: {recommendation['confidence']:.2f}")
            
            # Format comprehensive analysis result
            return f"""LEARNING_OPPORTUNITY_ANALYSIS:

TEACHING_INTENT: {teaching_intent['detected']} (Confidence: {teaching_intent['confidence']:.2f})
Reasoning: {teaching_intent['reasoning']}
Categories: {', '.join(teaching_intent.get('categories', []))}

KNOWLEDGE_GAP: {knowledge_gap['exists']} (Score: {knowledge_gap['score']:.2f})
Analysis: {knowledge_gap['reasoning']}

LEARNING_VALUE: {learning_value:.2f}/1.0
CONTENT_QUALITY: {content_quality:.2f}/1.0
SIMILARITY_SCORE: {similarity_score:.3f} (from existing knowledge)

RECOMMENDATION: {recommendation['action']}
Confidence: {recommendation['confidence']:.2f}
Reason: {recommendation['reason']}

NEXT_STEPS:
{recommendation['next_steps']}
"""
            
        except Exception as e:
            logger.error(f"Learning analysis error: {e}")
            return f"ANALYSIS_ERROR: Failed to analyze learning opportunity - {str(e)}"
    
    async def _detect_teaching_intent(self, message: str, context: str) -> Dict[str, Any]:
        """Detect if user has valuable information sharing intent using GPT-4o analysis"""
        try:
            # Use GPT-4o to analyze information sharing value with expanded criteria
            analysis_prompt = f"""
Analyze this message to determine if it contains valuable information that should be learned and remembered for better personalization and context understanding.

Message: "{message}"
Context: "{context}"

EXPANDED LEARNING CRITERIA - Look for ANY of these valuable information types:

🏢 **BUSINESS/COMPANY CONTEXT:**
- Company description, services, products
- Industry, market, business model
- Team size, structure, departments
- Business processes, workflows
- Company goals, values, culture

👤 **PERSONAL/PROFESSIONAL CONTEXT:**  
- User's role, job title, responsibilities
- Skills, expertise, experience level
- Work environment, tools used
- Professional goals, career focus
- Personal preferences, working style

🎯 **DOMAIN KNOWLEDGE:**
- Industry insights, expertise sharing
- Technical knowledge, best practices
- Process descriptions, methodologies
- Tool recommendations, experiences
- Problem-solving approaches

📊 **SITUATIONAL CONTEXT:**
- Current projects, initiatives
- Challenges, pain points, obstacles
- Requirements, specifications, needs
- Timelines, deadlines, constraints
- Success metrics, evaluation criteria

🔄 **OPERATIONAL INFORMATION:**
- How they work, daily routines
- Communication preferences
- Decision-making processes  
- Resource availability, budgets
- Vendor relationships, partnerships

❌ **NOT VALUABLE (Skip learning):**
- Pure greetings without context
- Generic thank you messages
- Simple yes/no responses
- Casual small talk
- Repetitive information already known

EVALUATION FRAMEWORK:
- **High Value**: Business context, domain expertise, operational details
- **Medium Value**: Personal preferences, tool usage, process insights
- **Low Value**: Casual conversation, generic responses

Respond with ONLY a JSON object in this exact format:
{{
    "has_information_value": true/false,
    "value_category": "business_context|personal_context|domain_knowledge|situational_context|operational_info|low_value",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of why this information is/isn't valuable for future interactions",
    "information_types": ["company_description", "role_info", "process_description", ...],
    "is_business_info": true/false,
    "is_personal_context": true/false,
    "is_domain_expertise": true/false,
    "personalization_value": 0.0-1.0,
    "context_building_value": 0.0-1.0,
    "future_utility": "high|medium|low"
}}

Information types can include: "company_description", "role_info", "industry_insight", "process_description", "tool_usage", "preferences", "goals", "challenges", "expertise_sharing", "operational_detail", "team_structure", "business_model", etc.

IMPORTANT: Focus on information VALUE for building better context and personalization, not just traditional "teaching" scenarios.
"""
            
            # Use GPT-4o for better analysis
            if self.llm_provider.lower() == "anthropic":
                import anthropic
                
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # Use latest Claude model
                    max_tokens=500,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": analysis_prompt}
                    ]
                )
                result_text = response.content[0].text.strip()
                
            else:
                from openai import OpenAI
                import os
                
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model="gpt-4o",  # Use GPT-4o for better analysis
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing information value for personalization and context building. Always respond with valid JSON only."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_tokens=500
                )
                result_text = response.choices[0].message.content.strip()
            
            # Parse the LLM response (result_text is already set above)
            
            # Clean up the response to extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            import json
            import re
            
            # Try to extract JSON from the response with better error handling
            llm_analysis = None
            try:
                # First try to parse as direct JSON
                llm_analysis = json.loads(result_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from text response
                json_match = re.search(r'\{[^}]*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        llm_analysis = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            # If we still don't have valid JSON, create a default response
            if not llm_analysis:
                logger.warning(f"Could not parse LLM information value analysis as JSON: {result_text[:200]}...")
                llm_analysis = {
                    "has_information_value": False,
                    "value_category": "low_value",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse LLM analysis response",
                    "information_types": [],
                    "is_business_info": False,
                    "is_personal_context": False,
                    "is_domain_expertise": False,
                    "personalization_value": 0.0,
                    "context_building_value": 0.0,
                    "future_utility": "low"
                }
            
            # Map the new analysis format to the legacy format for compatibility
            return {
                "detected": llm_analysis.get("has_information_value", False),
                "confidence": llm_analysis.get("confidence", 0.5),
                "reasoning": llm_analysis.get("reasoning", ""),
                "indicators": llm_analysis.get("information_types", []),
                "categories": [llm_analysis.get("value_category", "low_value")],
                "is_organizational_info": llm_analysis.get("is_business_info", False),
                "is_factual_content": llm_analysis.get("has_information_value", False),
                "knowledge_type": llm_analysis.get("value_category", "general"),
                # Enhanced metadata
                "personalization_value": llm_analysis.get("personalization_value", 0.0),
                "context_building_value": llm_analysis.get("context_building_value", 0.0),
                "future_utility": llm_analysis.get("future_utility", "low"),
                "is_personal_context": llm_analysis.get("is_personal_context", False),
                "is_domain_expertise": llm_analysis.get("is_domain_expertise", False)
            }
            
        except Exception as e:
            logger.error(f"Error in LLM information value detection: {str(e)}")
            # Fallback to basic detection
            return self._fallback_information_detection(message)
    
    def _fallback_information_detection(self, message: str) -> Dict[str, Any]:
        """Fallback information value detection if LLM call fails"""
        
        # Expanded patterns for valuable information
        business_patterns = [
            "công ty", "company", "business", "startup", "doanh nghiệp", "dịch vụ", "service",
            "sản phẩm", "product", "khách hàng", "customer", "client", "thị trường", "market",
            "ngành", "industry", "chuyên", "specialize", "làm", "do", "cung cấp", "provide"
        ]
        
        personal_patterns = [
            "tôi là", "i am", "i work", "my role", "my job", "responsibility", "responsible",
            "nhiệm vụ", "vai trò", "chức vụ", "position", "experience", "kinh nghiệm",
            "skill", "kỹ năng", "good at", "giỏi", "know how", "biết cách"
        ]
        
        process_patterns = [
            "process", "quy trình", "workflow", "how we", "cách chúng tôi", "procedure",
            "thủ tục", "method", "phương pháp", "approach", "cách tiếp cận", "strategy",
            "chiến lược", "plan", "kế hoạch", "system", "hệ thống"
        ]
        
        goal_patterns = [
            "goal", "mục tiêu", "want to", "muốn", "need to", "cần", "looking for",
            "tìm kiếm", "hope to", "hy vọng", "plan to", "dự định", "objective",
            "target", "aim", "aim to"
        ]
        
        message_lower = message.lower()
        
        # Score different types of valuable information
        business_score = len([p for p in business_patterns if p in message_lower])
        personal_score = len([p for p in personal_patterns if p in message_lower])  
        process_score = len([p for p in process_patterns if p in message_lower])
        goal_score = len([p for p in goal_patterns if p in message_lower])
        
        total_score = business_score + personal_score + process_score + goal_score
        
        # Determine information value
        has_value = total_score > 0
        confidence = min(0.9, total_score * 0.2) if has_value else 0.3
        
        # Determine category
        category = "low_value"
        if business_score > 0:
            category = "business_context"
        elif personal_score > 0:
            category = "personal_context"
        elif process_score > 0:
            category = "operational_info"
        elif goal_score > 0:
            category = "situational_context"
        
        indicators = []
        if business_score > 0:
            indicators.extend(["business_info", "company_description"])
        if personal_score > 0:
            indicators.extend(["personal_context", "role_info"])
        if process_score > 0:
            indicators.extend(["process_description", "operational_detail"])
        if goal_score > 0:
            indicators.extend(["goals", "objectives"])
        
        return {
            "detected": has_value,
            "confidence": confidence,
            "reasoning": f"Pattern-based detection found {total_score} information indicators: Business({business_score}), Personal({personal_score}), Process({process_score}), Goals({goal_score})" if has_value else "No valuable information patterns detected",
            "indicators": indicators,
            "categories": [category],
            "is_organizational_info": business_score > 0,
            "is_factual_content": has_value,
            "knowledge_type": category,
            # Enhanced metadata
            "personalization_value": confidence,
            "context_building_value": confidence,
            "future_utility": "high" if total_score >= 3 else "medium" if total_score >= 2 else "low",
            "is_personal_context": personal_score > 0,
            "is_domain_expertise": process_score > 0
        }
    
    def _detect_knowledge_gap(self, message: str, search_results: str) -> Dict[str, Any]:
        """Detect if there's a knowledge gap that learning would fill"""
        if "No existing knowledge found" in search_results:
            return {
                "exists": True,
                "score": 1.0,
                "reasoning": "No existing knowledge found - clear knowledge gap"
            }
        
        # Extract similarity scores from search results
        similarities = []
        lines = search_results.split('\n')
        for line in lines:
            if "Similarity:" in line:
                try:
                    similarity = float(line.split("Similarity:")[1].split(")")[0].strip())
                    similarities.append(similarity)
                except:
                    continue
        
        if similarities:
            max_similarity = max(similarities)
            gap_score = 1.0 - max_similarity  # Higher gap when similarity is low
            exists = gap_score > 0.3  # Knowledge gap exists if similarity < 0.7
            
            return {
                "exists": exists,
                "score": gap_score,
                "reasoning": f"Max similarity {max_similarity:.3f} indicates {'significant' if exists else 'minimal'} knowledge gap"
            }
        
        return {
            "exists": False,
            "score": 0.0,
            "reasoning": "Unable to determine knowledge gap from search results"
        }
    
    def _assess_learning_value(self, message: str) -> float:
        """Assess the learning value of the message content"""
        value_indicators = {
            "specific facts": 0.3,
            "company info": 0.4,
            "procedures": 0.4,
            "definitions": 0.3,
            "instructions": 0.4,
            "policies": 0.4,
            "contact info": 0.3,
            "dates/numbers": 0.2
        }
        
        message_lower = message.lower()
        total_value = 0.0
        
        # Check for company/organizational content
        if any(word in message_lower for word in ["company", "organization", "team", "department"]):
            total_value += 0.3
        
        # Check for factual content
        if any(word in message_lower for word in ["is", "are", "has", "have", "contains", "includes"]):
            total_value += 0.2
        
        # Check for instructional content
        if any(word in message_lower for word in ["should", "must", "need to", "have to", "always", "never"]):
            total_value += 0.3
        
        # Length factor (longer messages often have more value)
        length_factor = min(0.2, len(message.split()) / 50)  # Up to 0.2 for 50+ words
        total_value += length_factor
        
        return min(1.0, total_value)
    
    def _assess_content_quality(self, message: str) -> float:
        """Assess the quality and clarity of the message content"""
        # Basic quality indicators
        word_count = len(message.split())
        sentence_count = len([s for s in message.split('.') if s.strip()])
        
        quality_score = 0.0
        
        # Word count scoring
        if word_count >= 10:
            quality_score += 0.3
        if word_count >= 20:
            quality_score += 0.2
        
        # Sentence structure scoring
        if sentence_count >= 2:
            quality_score += 0.2
        
        # Completeness scoring
        if message.strip().endswith('.') or message.strip().endswith('!'):
            quality_score += 0.1
        
        # Specificity scoring (presence of specific terms)
        specific_terms = ['specific', 'exactly', 'precisely', 'details', 'information']
        if any(term in message.lower() for term in specific_terms):
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def _extract_similarity_from_search(self, search_results: str) -> float:
        """Extract the maximum similarity score from search results"""
        if "Max similarity:" in search_results:
            try:
                similarity_line = [line for line in search_results.split('\n') if "Max similarity:" in line][0]
                similarity = float(similarity_line.split("Max similarity:")[1].split(")")[0].strip())
                return similarity
            except:
                pass
        
        return 0.0
    
    def _generate_learning_recommendation(self, teaching_intent: Dict, knowledge_gap: Dict, learning_value: float, content_quality: float, similarity_score: float) -> Dict[str, Any]:
        """Generate overall learning recommendation based on expanded information value analysis"""
        
        # Extract enhanced information from the analysis
        detected_info = teaching_intent['detected']
        confidence = teaching_intent['confidence']
        personalization_value = teaching_intent.get('personalization_value', 0.0)
        context_building_value = teaching_intent.get('context_building_value', 0.0)
        future_utility = teaching_intent.get('future_utility', 'low')
        is_business_info = teaching_intent.get('is_organizational_info', False)
        is_personal_context = teaching_intent.get('is_personal_context', False)
        is_domain_expertise = teaching_intent.get('is_domain_expertise', False)
        
        # EXPANDED LEARNING DECISION LOGIC
        
        # Strong indicators for learning - HIGH PRIORITY
        if detected_info and confidence > 0.6:
            # Business context is almost always valuable
            if is_business_info or personalization_value >= 0.7 or context_building_value >= 0.7:
                return {
                    "action": "SHOULD_LEARN",
                    "confidence": 0.9,
                    "reason": f"High-value information detected: Business({is_business_info}), Personalization({personalization_value:.2f}), Context({context_building_value:.2f})",
                    "next_steps": "Create learning decision for human approval - this information will improve future interactions"
                }
            
            # Domain expertise and personal context are also valuable
            if is_domain_expertise or is_personal_context:
                if knowledge_gap['exists'] and knowledge_gap['score'] > 0.3:
                    return {
                        "action": "SHOULD_LEARN",
                        "confidence": 0.85,
                        "reason": f"Valuable context detected: Domain expertise({is_domain_expertise}), Personal context({is_personal_context}) with knowledge gap({knowledge_gap['score']:.2f})",
                        "next_steps": "Create learning decision for human approval - this context will enhance personalization"
                    }
        
        # Medium priority - consider learning with moderate confidence
        if detected_info and confidence > 0.4:
            # If there's a clear knowledge gap and decent quality
            if knowledge_gap['exists'] and learning_value >= 0.4 and content_quality >= 0.3:
                return {
                    "action": "MAYBE_LEARN",
                    "confidence": 0.7,
                    "reason": f"Moderate value information with knowledge gap: Value({learning_value:.2f}), Quality({content_quality:.2f}), Future utility({future_utility})",
                    "next_steps": "Present to human for decision - information has moderate learning value"
                }
            
            # High-value information even without perfect quality
            if personalization_value >= 0.5 or context_building_value >= 0.5:
                return {
                    "action": "MAYBE_LEARN",
                    "confidence": 0.75,
                    "reason": f"Good personalization/context building value: Personalization({personalization_value:.2f}), Context({context_building_value:.2f})",
                    "next_steps": "Present to human for decision - will improve understanding of user context"
                }
        
        # Handle potential updates to existing knowledge
        if similarity_score > 0.7 and detected_info and confidence > 0.3:
            return {
                "action": "MAYBE_UPDATE",
                "confidence": 0.6,
                "reason": f"Similar content exists (similarity: {similarity_score:.3f}) but contains new information with confidence {confidence:.2f}",
                "next_steps": "Consider updating existing knowledge instead of creating new - presents options to human"
            }
        
        # Special case: Even low confidence business/personal info might be worth asking about
        if (is_business_info or is_personal_context) and confidence > 0.3:
            return {
                "action": "MAYBE_LEARN",
                "confidence": 0.6,
                "reason": f"Business or personal context detected with moderate confidence ({confidence:.2f}) - valuable for personalization even if not perfectly clear",
                "next_steps": "Ask human to decide - business/personal context is often valuable even when uncertain"
            }
        
        # Low learning indicators - skip learning
        return {
            "action": "NO_LEARN",
            "confidence": 0.8,
            "reason": f"Insufficient information value: Confidence({confidence:.2f}), Personalization({personalization_value:.2f}), Context({context_building_value:.2f}), Future utility({future_utility})",
            "next_steps": "Continue conversation without learning - information doesn't meet learning thresholds"
        }

class HumanLearningTool:
    """Tool for LLM to request human decisions about learning"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown", llm_provider: str = "openai", model: str = "gpt-4o"):
        self.user_id = user_id
        self.org_id = org_id
        self.llm_provider = llm_provider
        self.model = model or "gpt-4o"  # Default to GPT-4o
        self.name = "human_learning"
    
    def request_learning_decision(self, decision_type: str, context: str, options: List[str], additional_info: str = "", thread_id: str = None) -> str:
        """
        LLM calls this to request a human decision about learning
        
        Args:
            decision_type: Type of decision needed
            context: Context for the decision
            options: List of available options
            additional_info: Additional information for decision making
            thread_id: Optional thread ID for context
            
        Returns:
            Decision request confirmation with decision ID
        """
        try:
            # Generate unique decision ID
            decision_id = f"learning_decision_{uuid.uuid4().hex[:8]}"
            
            # Store decision request globally
            PENDING_LEARNING_DECISIONS[decision_id] = {
                "id": decision_id,
                "type": decision_type,
                "context": context,
                "options": options,
                "additional_info": additional_info,
                "user_id": self.user_id,
                "org_id": self.org_id,
                "thread_id": thread_id,
                "status": "PENDING",
                "created_at": datetime.now().isoformat(),
                "human_choice": None,
                "completed_at": None
            }
            
            logger.info(f"Created learning decision request: {decision_id} for user {self.user_id}")
            
            return f"""LEARNING_DECISION_REQUESTED:

Decision ID: {decision_id}
Type: {decision_type}
Context: {context}
Options: {', '.join(options)}

STATUS: Pending human decision
NEXT_STEPS: 
- Human will be presented with these options
- Decision will be processed when human chooses
- Knowledge will only be saved if human approves

Additional Info: {additional_info}
"""
            
        except Exception as e:
            logger.error(f"Error creating learning decision: {e}")
            return f"DECISION_ERROR: Failed to create learning decision - {str(e)}"

class KnowledgePreviewTool:
    """Tool for LLM to preview what knowledge would be saved"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown"):
        self.user_id = user_id
        self.org_id = org_id
        self.name = "knowledge_preview"
    
    def preview_knowledge_save(self, user_message: str, ai_response: str, 
                             save_format: str = "conversation") -> str:
        """
        LLM calls this to preview what would be saved as knowledge
        
        Args:
            user_message: The user's original message
            ai_response: The AI's response/synthesis
            save_format: Format to save in ("conversation", "synthesis", "summary")
            
        Returns:
            Preview of what would be saved
        """
        try:
            logger.info(f"Previewing knowledge save in format: {save_format}")
            
            preview_content = ""
            
            if save_format == "conversation":
                preview_content = f"User: {user_message}\n\nAI: {ai_response}"
            elif save_format == "synthesis":
                preview_content = ai_response
            elif save_format == "summary":
                # Create a brief summary
                summary = f"Summary: {user_message[:100]}..." if len(user_message) > 100 else user_message
                preview_content = f"{summary}\n\nAI Analysis: {ai_response[:200]}..." if len(ai_response) > 200 else ai_response
            elif save_format == "all":
                preview_content = f"""=== CONVERSATION FORMAT ===
User: {user_message}

AI: {ai_response}

=== SYNTHESIS FORMAT ===
{ai_response}

=== SUMMARY FORMAT ===
Summary: {user_message[:100]}...
AI Analysis: {ai_response[:200]}..."""
            
            return f"""KNOWLEDGE_SAVE_PREVIEW:

Format: {save_format}
Content Length: {len(preview_content)} characters
User ID: {self.user_id}
Organization: {self.org_id}

--- CONTENT PREVIEW ---
{preview_content}
--- END PREVIEW ---

METADATA:
- Source: Interactive Learning
- Categories: ["teaching_intent", "human_approved"]
- Timestamp: {datetime.now().isoformat()}

NOTE: This is only a preview. Actual saving requires human approval.
"""
            
        except Exception as e:
            logger.error(f"Preview error: {e}")
            return f"PREVIEW_ERROR: Failed to generate preview - {str(e)}"
    
    def get_tool_description(self) -> Dict[str, Any]:
        """Get tool description for LLM function calling"""
        return {
            "name": "preview_knowledge_save",
            "description": "Preview what knowledge would be saved",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The user's original message"
                    },
                    "ai_response": {
                        "type": "string",
                        "description": "The AI's response/synthesis"
                    },
                    "save_format": {
                        "type": "string",
                        "enum": ["conversation", "synthesis", "summary", "all"],
                        "description": "Format to save knowledge in",
                        "default": "conversation"
                    }
                },
                "required": ["user_message", "ai_response"]
            }
        }


class KnowledgeSaveTool:
    """Tool for LLM to actually save knowledge (REQUIRES HUMAN APPROVAL)"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown"):
        self.user_id = user_id
        self.org_id = org_id
        self.name = "knowledge_save"
    
    async def save_knowledge(self, content: str, title: str = "", categories: List[str] = None, 
                      thread_id: str = None, decision_id: str = None) -> str:
        """
        LLM calls this to save knowledge (REQUIRES HUMAN APPROVAL via decision_id)
        
        Args:
            content: The knowledge content to save
            title: Optional title for the knowledge
            categories: Categories to tag the knowledge with
            thread_id: Thread ID for conversation context
            decision_id: Reference to human decision (REQUIRED)
            
        Returns:
            Save status and result
        """
        try:
            logger.info(f"KnowledgeSaveTool: [STEP 1] Starting save for user {self.user_id}")
            
            # 🚨 CRITICAL SECURITY FIX: REQUIRE decision_id for ALL saves
            if not decision_id:
                error_msg = "SAVE_REJECTED: decision_id is REQUIRED for all knowledge saves. Use request_learning_decision first to get human approval."
                logger.error(f"KnowledgeSaveTool: [SECURITY] {error_msg}")
                return error_msg
            
            # ALWAYS validate human approval
            logger.info(f"KnowledgeSaveTool: [STEP 2] Checking human approval for decision {decision_id}")
            approval_status = self._check_human_approval(decision_id)
            if not approval_status["approved"]:
                error_msg = f"SAVE_REJECTED: {approval_status['reason']}"
                logger.error(f"KnowledgeSaveTool: [SECURITY] {error_msg}")
                return error_msg
            
            logger.info(f"KnowledgeSaveTool: [STEP 3] ✅ Human approval confirmed, preparing save parameters")
            # Prepare save parameters
            save_params = {
                "content": content,
                "title": title or f"Knowledge from {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "categories": categories or ["teaching_intent", "human_approved"],
                "user_id": self.user_id,
                "org_id": self.org_id,
                "thread_id": thread_id,
                "metadata": {
                    "source": "interactive_learning",
                    "decision_id": decision_id,
                    "saved_at": datetime.now().isoformat(),
                    "human_approved": True  # Explicitly mark as human approved
                }
            }
            
            logger.info(f"KnowledgeSaveTool: [STEP 4] Calling _perform_knowledge_save")
            # Perform the actual save
            save_result = await self._perform_knowledge_save(save_params)
            
            logger.info(f"KnowledgeSaveTool: [STEP 5] Save completed with result: {save_result}")
            # Log save attempt for debugging
            logger.info(f"✅ HUMAN APPROVED KNOWLEDGE SAVE - Success: {save_result['success']}, "
                       f"Title: '{title[:30]}...', Content Length: {len(content)}, "
                       f"Decision ID: {decision_id}")
            
            if save_result["success"]:
                return f"""
✅ KNOWLEDGE_SAVED_SUCCESSFULLY:
- Knowledge ID: {save_result.get('knowledge_id', 'unknown')}
- Title: {save_params['title']}
- Categories: {', '.join(save_params['categories'])}
- Content Length: {len(content)} characters
- Vectors Created: {save_result.get('vectors_created', 1)}
- Human Approved: ✅ YES (Decision ID: {decision_id})

IMPACT:
- This knowledge is now searchable for future questions
- It will be considered in similarity calculations
- It contributes to the organization's knowledge base

NEXT_STEPS:
- Continue the conversation naturally
- The saved knowledge will be used to improve future responses
                """.strip()
            else:
                return f"SAVE_FAILED: {save_result.get('error', 'Unknown error occurred')}"
                
        except Exception as e:
            logger.error(f"KnowledgeSaveTool error at step: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"ERROR: Failed to save knowledge - {str(e)}"
    
    def _check_human_approval(self, decision_id: str) -> Dict[str, Any]:
        """Check if human has approved the save"""
        if decision_id not in PENDING_LEARNING_DECISIONS:
            return {"approved": False, "reason": "Decision not found - invalid decision_id"}
        
        decision = PENDING_LEARNING_DECISIONS[decision_id]
        
        if decision["status"] != "COMPLETED":
            return {"approved": False, "reason": "Decision not completed - waiting for human choice"}
        
        human_choice = decision.get("human_choice", "")
        
        # Check if choice indicates approval
        approval_choices = ["save_new", "update_existing", "yes", "approve", "proceed", "save"]
        approved = any(choice in human_choice.lower() for choice in approval_choices)
        
        return {
            "approved": approved, 
            "reason": "Human approved the knowledge save" if approved else "Human chose not to save knowledge"
        }
    
    async def _perform_knowledge_save(self, save_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual knowledge saving using the correct save_knowledge function like AVA does"""
        try:
            # Use the same save_knowledge function that AVA uses for Pinecone
            from pccontroller import save_knowledge
            
            logger.info(f"✅ Calling pccontroller.save_knowledge for HUMAN APPROVED decision-based save")
            
            # Call save_knowledge with the same parameters AVA uses
            result = await save_knowledge(
                input=save_params["content"],
                user_id=save_params["user_id"],
                org_id=save_params["org_id"],
                title=save_params["title"],
                thread_id=save_params.get("thread_id"),
                topic=save_params.get("title", "interactive_learning"),
                categories=save_params.get("categories", ["teaching_intent", "human_approved"]),
                ttl_days=365  # 365 days TTL like AVA
            )
            
            logger.info(f"pccontroller.save_knowledge result: {result}")
            
            if result and result.get("success"):
                vector_id = result.get("vector_id")
                logger.info(f"✅ KNOWLEDGE SAVED SUCCESSFULLY TO PINECONE: ID: {vector_id}")
                return {
                    "success": True,
                    "knowledge_id": vector_id,
                    "vectors_created": 1,
                    "namespace": result.get("namespace", "conversation"),
                    "created_at": result.get("created_at")
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "save_knowledge returned None"
                logger.error(f"Knowledge save failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
            
        except Exception as e:
            logger.error(f"Knowledge save failed with error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tool_description(self) -> Dict[str, Any]:
        """Get tool description for LLM function calling"""
        return {
            "name": "save_knowledge",
            "description": "Actually save knowledge (REQUIRES HUMAN APPROVAL via decision_id)",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The knowledge content to save"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the knowledge",
                        "default": ""
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categories to tag the knowledge with",
                        "default": ["teaching_intent", "human_approved"]
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID for conversation context",
                        "default": None
                    },
                    "decision_id": {
                        "type": "string",
                        "description": "Reference to human decision (REQUIRED FOR ALL SAVES)",
                        # 🚨 CRITICAL: Remove default None and make it required
                    }
                },
                "required": ["content", "decision_id"]  # 🚨 MAKE decision_id REQUIRED
            }
        }


class LearningToolsFactory:
    """Factory for creating and managing learning tools"""
    
    @staticmethod
    def create_learning_tools(user_id: str = "unknown", org_id: str = "unknown", 
                            llm_provider: str = "openai", model: str = None) -> List[Any]:
        """Create all learning tools for a user/org"""
        return [
            LearningSearchTool(user_id, org_id),
            LearningAnalysisTool(user_id, org_id, llm_provider, model),
            HumanLearningTool(user_id, org_id),
            KnowledgePreviewTool(user_id, org_id),
            KnowledgeSaveTool(user_id, org_id)
        ]
    
    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """Get LLM tool definitions for all learning tools"""
        return [
            {
                "name": "search_learning_context",
                "description": "Search for existing knowledge relevant to potential learning opportunities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in existing knowledge"
                        },
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "deep", "exhaustive"],
                            "description": "How thorough the search should be",
                            "default": "basic"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_learning_opportunity",
                "description": "Analyze whether something should be learned from user input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {
                            "type": "string",
                            "description": "The user's message to analyze"
                        },
                        "conversation_context": {
                            "type": "string",
                            "description": "Previous conversation context",
                            "default": ""
                        },
                        "search_results": {
                            "type": "string",
                            "description": "Results from learning search tool",
                            "default": ""
                        }
                    },
                    "required": ["user_message"]
                }
            },
            {
                "name": "request_learning_decision",
                "description": "Ask human for decision about learning action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision_type": {
                            "type": "string",
                            "enum": ["save_new", "update_existing", "merge_knowledge", "skip_learning"],
                            "description": "Type of decision needed"
                        },
                        "context": {
                            "type": "string",
                            "description": "Context about what the decision is for"
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Available options for human to choose"
                        },
                        "additional_info": {
                            "type": "string",
                            "description": "Additional info to help with decision",
                            "default": ""
                        }
                    },
                    "required": ["decision_type", "context", "options"]
                }
            },
            {
                "name": "preview_knowledge_save",
                "description": "Preview what knowledge would be saved",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {
                            "type": "string",
                            "description": "The user's original message"
                        },
                        "ai_response": {
                            "type": "string",
                            "description": "The AI's response/synthesis"
                        },
                        "save_format": {
                            "type": "string",
                            "enum": ["conversation", "synthesis", "summary", "all"],
                            "description": "Format to save knowledge in",
                            "default": "conversation"
                        }
                    },
                    "required": ["user_message", "ai_response"]
                }
            },
            {
                "name": "save_knowledge",
                "description": "Actually save knowledge (REQUIRES HUMAN APPROVAL via decision_id)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The knowledge content to save"
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title for the knowledge",
                            "default": ""
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Categories to tag the knowledge with",
                            "default": ["teaching_intent", "human_approved"]
                        },
                        "thread_id": {
                            "type": "string",
                            "description": "Thread ID for conversation context",
                            "default": None
                        },
                        "decision_id": {
                            "type": "string",
                            "description": "Reference to human decision (REQUIRED FOR ALL SAVES)",
                            "default": None
                        }
                    },
                    "required": ["content", "decision_id"]  # 🚨 MAKE decision_id REQUIRED
                }
            }
        ]


# Utility functions for decision management
def get_pending_decisions(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get pending learning decisions, optionally filtered by user_id"""
    if user_id:
        # Filter by user_id
        return {
            decision_id: decision 
            for decision_id, decision in PENDING_LEARNING_DECISIONS.items()
            if decision.get("user_id") == user_id and decision.get("status") == "PENDING"
        }
    else:
        # Return all pending decisions
        return {
            decision_id: decision 
            for decision_id, decision in PENDING_LEARNING_DECISIONS.items()
            if decision.get("status") == "PENDING"
        }


async def complete_learning_decision(decision_id: str, human_choice: str) -> Dict[str, Any]:
    """Complete a learning decision with human choice and trigger knowledge saving if approved"""
    if decision_id not in PENDING_LEARNING_DECISIONS:
        return {"success": False, "error": "Decision not found"}
    
    decision = PENDING_LEARNING_DECISIONS[decision_id]
    decision["status"] = "COMPLETED"
    decision["human_choice"] = human_choice
    decision["completed_at"] = datetime.now().isoformat()
    
    # If human approved saving, trigger the actual knowledge saving process
    approval_choices = ["save_new", "Save as new knowledge", "yes", "approve", "proceed"]
    should_save = any(choice.lower() in human_choice.lower() for choice in approval_choices)
    
    if should_save:
        try:
            # Extract information from the decision context
            context = decision.get("context", "")
            user_id = decision.get("user_id", "unknown")
            org_id = decision.get("org_id", "unknown")
            
            # Extract the original user message from context
            if "Teaching content detected:" in context:
                user_message = context.replace("Teaching content detected:", "").strip()
            else:
                user_message = context
            
            # Get the AI response that was generated for this user message
            # This should include AI's understanding and synthesis
            ai_response = decision.get("ai_response", "") or decision.get("ai_synthesis", "")
            
            # If no AI response available, generate one using LLM synthesis
            if not ai_response:
                logger.info(f"No AI response found for decision {decision_id}, generating AI synthesis...")
                
                try:
                    # Generate AI synthesis of the user's teaching content
                    synthesis_prompt = f"""
You are Ami, a no-code AI agent builder that helps people transform their imagination into practical AI agents. A human just shared this valuable information with you:

"{user_message}"

Please provide a thoughtful synthesis that demonstrates your deep understanding. Structure your response as:

**What I learned:** [Summarize what the human taught you]

**Why this matters:** [Explain how this information is valuable and what insights it provides]

**Building connections:** [How this knowledge helps you better understand their world, needs, or the AI agents they might want to create]

**Future possibilities:** [How this information could influence the AI agent recommendations or capabilities you might suggest]

Write in Ami's voice - enthusiastic about possibilities, focused on turning imagination into reality, and genuinely excited about learning from humans. Keep it concise but meaningful.
                    """
                    
                    # Use GPT-4o for better synthesis quality
                    from openai_tool import OpenAITool
                    openai_tool = OpenAITool(model="gpt-4o")  # Advanced model for high-quality synthesis
                    # Use async-safe method
                    ai_response = await asyncio.to_thread(openai_tool.generate_response, synthesis_prompt)
                    
                    if ai_response and len(ai_response.strip()) > 20:
                        logger.info(f"Generated AI synthesis: {ai_response[:100]}...")
                        # Store the generated response in the decision for future reference
                        decision["ai_synthesis"] = ai_response
                    else:
                        ai_response = f"I appreciate you sharing this information about {user_message[:100]}. This knowledge will be valuable for helping others build AI agents."
                        logger.warning(f"LLM synthesis failed, using fallback response")
                        
                except Exception as e:
                    logger.error(f"Failed to generate AI synthesis: {e}")
                    ai_response = f"I understand and appreciate this information: {user_message[:100]}. This will help me assist others better."
                    
            else:
                logger.info(f"Found AI response for enhanced knowledge saving: {ai_response[:100]}...")
            
            # Use AVA-style multi-vector saving approach for richer knowledge storage
            from ava import AVA
            ava_instance = AVA()
            
            # Create a response-like structure for AVA's saving methods
            response_structure = {
                "message": ai_response,
                "metadata": {
                    "response_strategy": "LEARNING_DECISION",
                    "has_teaching_intent": True,
                    "is_priority_topic": True,
                    "should_save_knowledge": True
                }
            }
            
            # Use AVA's sophisticated multi-vector saving approach
            logger.info(f"Using AVA-style multi-vector knowledge saving for decision {decision_id}")
            
            try:
                # Save using AVA's _save_tool_knowledge_multiple method which saves:
                # 1. Combined Knowledge (User + AI format)
                # 2. AI Synthesis (enhanced AI understanding)  
                # 3. User message only (for reference)
                await ava_instance._save_tool_knowledge_multiple(
                    user_message=user_message,
                    ai_response=ai_response,
                    combined_content=f"User: {user_message}\n\nAI: {ai_response}",
                    user_id=user_id,
                    thread_id=decision.get("thread_id"),
                    priority_topic_name=decision.get("topic", "interactive_learning"),
                    categories=["teaching_intent", "human_approved", "interactive_learning", "multi_vector_save"],
                    response=response_structure,
                    org_id=org_id
                )
                
                save_result = {
                    "success": True,
                    "message": "Knowledge saved using multi-vector AVA approach",
                    "vectors_saved": 3,
                    "save_method": "ava_multi_vector"
                }
                
            except Exception as ava_error:
                logger.error(f"AVA multi-vector save failed, falling back to single save: {ava_error}")
                
                # Fallback to single enhanced save if AVA method fails
                save_tool = KnowledgeSaveTool(user_id=user_id, org_id=org_id)
                
                # Enhanced knowledge content with both user and AI
                knowledge_content = f"User: {user_message}\n\nAI Synthesis: {ai_response}"
                knowledge_title = f"Interactive Learning - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                knowledge_categories = ["teaching_intent", "human_approved", "interactive_learning", "enhanced_save"]
                
                # Call the actual save_knowledge function with enhanced content
                logger.info(f"Calling enhanced save_knowledge for decision {decision_id}")
                save_result = await save_tool.save_knowledge(
                    content=knowledge_content,
                    title=knowledge_title,
                    categories=knowledge_categories,
                    decision_id=decision_id
                )
            
            # Log the actual save result
            logger.info(f"Knowledge save result: {save_result}")
            
            # Add save results to decision record
            decision["knowledge_saved"] = True
            decision["save_result"] = save_result
            
            logger.info(f"Learning workflow completed successfully - Decision: {decision_id} | Knowledge saved to database")
            
            return {
                "success": True, 
                "decision": decision,
                "knowledge_saved": True,
                "save_status": save_result
            }
            
        except Exception as e:
            logger.error(f"Error saving knowledge after human approval: {e}")
            decision["knowledge_saved"] = False
            decision["save_error"] = str(e)
            
            return {
                "success": True,
                "decision": decision, 
                "knowledge_saved": False,
                "save_error": str(e)
            }
    else:
        # Human chose not to save
        decision["knowledge_saved"] = False
        return {"success": True, "decision": decision, "knowledge_saved": False}
    
    return {"success": True, "decision": decision}


def cleanup_expired_decisions():
    """Clean up expired learning decisions"""
    current_time = datetime.now()
    expired_ids = []
    
    for decision_id, decision in PENDING_LEARNING_DECISIONS.items():
        try:
            expiry_time = datetime.fromisoformat(decision["expires_at"])
            if current_time > expiry_time:
                expired_ids.append(decision_id)
        except:
            # If we can't parse expiry time, consider it expired
            expired_ids.append(decision_id)
    
    for decision_id in expired_ids:
        PENDING_LEARNING_DECISIONS[decision_id]["status"] = "EXPIRED"
    
    return len(expired_ids) 