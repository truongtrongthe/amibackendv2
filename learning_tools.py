"""
Interactive Learning Tools - Phase 1 Implementation

This module provides explicit learning tools that the LLM can call to make
the learning process interactive and human-guided, replacing the monolithic
hidden learning logic.
"""

import logging
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Global storage for pending learning decisions
PENDING_LEARNING_DECISIONS = {}


class LearningSearchTool:
    """Tool for LLM to search for learning-relevant knowledge"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown"):
        self.user_id = user_id
        self.org_id = org_id
        self.name = "learning_search"
    
    def search_learning_context(self, query: str, search_depth: str = "basic") -> str:
        """
        LLM calls this to search for existing knowledge relevant to learning
        
        Args:
            query: What to search for
            search_depth: "basic" (1 round), "deep" (2 rounds), "exhaustive" (3 rounds)
            
        Returns:
            Formatted search results for LLM to understand
        """
        try:
            logger.info(f"LearningSearchTool: Searching for '{query}' with depth '{search_depth}'")
            
            # Map search depth to exploration rounds
            depth_mapping = {
                "basic": 1,
                "deep": 2, 
                "exhaustive": 3
            }
            max_rounds = depth_mapping.get(search_depth, 1)
            
            # Use existing knowledge exploration logic (simplified for sync context)
            search_result = self._perform_knowledge_search_sync(query, max_rounds)
            
            # Format results for LLM
            similarity = search_result.get("similarity", 0.0)
            knowledge_context = search_result.get("knowledge_context", "")
            query_results = search_result.get("query_results", [])
            
            # Determine recommendation based on similarity
            if similarity >= 0.70:
                recommendation = "HIGH_SIMILARITY - Consider updating existing knowledge"
            elif similarity >= 0.35:
                recommendation = "MEDIUM_SIMILARITY - Human decision recommended"
            else:
                recommendation = "LOW_SIMILARITY - Safe to create new knowledge"
            
            # Log search results for debugging
            logger.info(f"LearningSearch Results - Query: '{query}', Similarity: {similarity:.2f}, "
                       f"Items Found: {len(query_results)}, Recommendation: {recommendation.split(' - ')[0]}")
            
            return f"""
LEARNING SEARCH RESULTS:
- Query: {query}
- Search Depth: {search_depth} ({max_rounds} rounds)
- Knowledge Items Found: {len(query_results)}
- Best Similarity Score: {similarity:.2f}
- Confidence Level: {self._calculate_confidence(similarity, len(query_results))}
- Recommendation: {recommendation}

EXISTING KNOWLEDGE PREVIEW:
{knowledge_context[:500] + "..." if len(knowledge_context) > 500 else knowledge_context}

SIMILARITY ANALYSIS:
- High (≥70%): Update existing knowledge
- Medium (35-70%): Human decision needed
- Low (<35%): Create new knowledge
            """.strip()
            
        except Exception as e:
            logger.error(f"LearningSearchTool error: {e}")
            return f"ERROR: Failed to search learning context - {str(e)}"
    
    def _perform_knowledge_search_sync(self, query: str, max_rounds: int) -> Dict[str, Any]:
        """Perform knowledge search synchronously (simplified for tool context)"""
        try:
            # For now, return a simplified search result to avoid async issues
            # This can be enhanced later with proper sync search implementation
            logger.info(f"Performing simplified search for: {query}")
            
            # Return mock result with low similarity to trigger learning
            return {
                "similarity": 0.1,  # Low similarity to encourage learning
                "knowledge_context": f"No existing knowledge found for '{query}'. This appears to be new information that could be valuable to learn.",
                "query_results": [],
                "queries": [query]
            }
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return {
                "similarity": 0.0,
                "knowledge_context": "",
                "query_results": [],
                "queries": [query]
            }

    async def _perform_knowledge_search(self, query: str, max_rounds: int) -> Dict[str, Any]:
        """Perform the actual knowledge search using existing exploration logic (async version)"""
        try:
            # Import and use existing exploration logic
            from curiosity import KnowledgeExplorer
            from learning_support import LearningSupport
            
            # Create knowledge explorer
            explorer = KnowledgeExplorer(
                graph_version_id="default",
                support_module=LearningSupport(None)  # We'll handle this gracefully
            )
            
            # Perform exploration
            result = await explorer.explore(
                message=query,
                conversation_context="",
                user_id=self.user_id,
                thread_id=None,
                max_rounds=max_rounds,
                org_id=self.org_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return {
                "similarity": 0.0,
                "knowledge_context": "",
                "query_results": [],
                "queries": [query]
            }
    
    def _calculate_confidence(self, similarity: float, result_count: int) -> str:
        """Calculate confidence level based on similarity and result count"""
        if similarity >= 0.70 and result_count >= 3:
            return "HIGH"
        elif similarity >= 0.35 and result_count >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_tool_description(self) -> Dict[str, Any]:
        """Get tool description for LLM function calling"""
        return {
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
        }


class LearningAnalysisTool:
    """Tool for LLM to analyze learning opportunities"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown"):
        self.user_id = user_id
        self.org_id = org_id
        self.name = "learning_analysis"
    
    def analyze_learning_opportunity(self, user_message: str, conversation_context: str = "", search_results: str = "") -> str:
        """
        LLM calls this to analyze if something should be learned
        
        Args:
            user_message: The user's message to analyze
            conversation_context: Previous conversation context
            search_results: Results from learning search tool
            
        Returns:
            Detailed analysis of learning opportunity
        """
        try:
            logger.info(f"LearningAnalysisTool: Analyzing message '{user_message[:50]}...'")
            
            # Analyze different aspects
            teaching_intent = self._detect_teaching_intent(user_message, conversation_context)
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
            
            return f"""
LEARNING OPPORTUNITY ANALYSIS:

TEACHING INTENT DETECTION (LLM-POWERED):
- Has Teaching Intent: {teaching_intent['detected']}
- Confidence: {teaching_intent['confidence']:.2f}
- LLM Reasoning: {reasoning}
- Categories Detected: {', '.join(teaching_intent['indicators'])}

KNOWLEDGE GAP ANALYSIS:
- Knowledge Gap Detected: {knowledge_gap['detected']}
- Gap Type: {knowledge_gap['type']}
- Significance: {knowledge_gap['significance']}

LEARNING VALUE ASSESSMENT:
- Educational Value: {learning_value['educational_value']:.2f}
- Practical Value: {learning_value['practical_value']:.2f}
- Uniqueness: {learning_value['uniqueness']:.2f}

CONTENT QUALITY:
- Clarity: {content_quality['clarity']:.2f}
- Completeness: {content_quality['completeness']:.2f}
- Accuracy Indicators: {content_quality['accuracy']}

SIMILARITY TO EXISTING:
- Similarity Score: {similarity_score:.2f}
- Similarity Category: {self._categorize_similarity(similarity_score)}

OVERALL RECOMMENDATION:
- Action: {recommendation['action']}
- Confidence: {recommendation['confidence']:.2f}
- Reasoning: {recommendation['reasoning']}
- Next Steps: {recommendation['next_steps']}

NOTE: Teaching intent detection is now powered by LLM analysis for better accuracy and flexibility.
            """.strip()
            
        except Exception as e:
            logger.error(f"LearningAnalysisTool error: {e}")
            return f"ERROR: Failed to analyze learning opportunity - {str(e)}"
    
    def get_tool_description(self) -> Dict[str, Any]:
        """Get tool description for LLM function calling"""
        return {
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
        }
    
    def _detect_teaching_intent(self, message: str, context: str) -> Dict[str, Any]:
        """Detect if user has teaching intent using LLM analysis"""
        try:
            # Use LLM to analyze teaching intent instead of hardcoded patterns
            analysis_prompt = f"""
Analyze this message to determine if the user has teaching intent (wants to share knowledge, give instructions, or provide information that should be learned/remembered).

Message: "{message}"
Context: "{context}"

Please analyze if this message contains:
1. Teaching intent (sharing knowledge, giving instructions, providing facts)
2. Company/organizational information 
3. Task assignments or procedures
4. Factual information that should be remembered
5. Educational content

Respond with ONLY a JSON object in this exact format:
{{
    "has_teaching_intent": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "categories": ["category1", "category2", ...],
    "is_organizational_info": true/false,
    "is_factual_content": true/false
}}

Categories can include: "teaching", "instruction", "company_info", "task_assignment", "factual_sharing", "procedure", "explanation"
"""

            # Use OpenAI to analyze the message
            from openai import OpenAI
            import os
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for analysis
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing teaching intent in messages. Always respond with valid JSON only."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=200
            )
            
            # Parse the LLM response
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response to extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            import json
            llm_analysis = json.loads(result_text)
            
            # Convert LLM analysis to our expected format
            detected = llm_analysis.get("has_teaching_intent", False)
            confidence = float(llm_analysis.get("confidence", 0.0))
            categories = llm_analysis.get("categories", [])
            reasoning = llm_analysis.get("reasoning", "")
            
            # Boost confidence for organizational info or factual content
            if llm_analysis.get("is_organizational_info", False):
                confidence = min(confidence + 0.3, 1.0)
                categories.append("organizational_info")
            
            if llm_analysis.get("is_factual_content", False):
                confidence = min(confidence + 0.2, 1.0)
                categories.append("factual_content")
            
            logger.info(f"LLM Teaching Intent Analysis - Detected: {detected}, Confidence: {confidence:.2f}, "
                       f"Categories: {categories}, Reasoning: {reasoning}")
            
            return {
                "detected": detected,
                "confidence": confidence,
                "indicators": categories,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"LLM teaching intent analysis failed: {e}")
            # Fallback to simple heuristics if LLM analysis fails
            return self._fallback_teaching_intent_detection(message, context)
    
    def _fallback_teaching_intent_detection(self, message: str, context: str) -> Dict[str, Any]:
        """Fallback teaching intent detection using simple heuristics"""
        indicators = []
        message_lower = message.lower()
        
        # Strong organizational indicators (specific to company/business context)
        if any(org in message_lower for org in ["company", "công ty", "organization", "tổ chức", "our company", "của chúng ta"]):
            indicators.append("organizational_info")
        
        # Strong task/instruction indicators
        if any(task in message_lower for task in ["task", "nhiệm vụ", "job", "công việc", "need to do", "cần làm", "your task", "nhiệm vụ của"]):
            indicators.append("task_assignment")
        
        # Strong procedural indicators
        if any(proc in message_lower for proc in ["procedure", "process", "quy trình", "step", "bước", "how to", "cách"]):
            indicators.append("procedural_info")
        
        # Specific factual/informational indicators (avoid generic words)
        if any(info in message_lower for info in ["information about", "thông tin về", "data shows", "dữ liệu cho thấy", "revenue", "doanh thu", "employees", "nhân viên"]):
            indicators.append("factual_sharing")
        
        # Filter out casual conversation patterns
        casual_patterns = ["how are you", "weather", "thời tiết", "hello", "xin chào", "thanks", "cảm ơn", "good morning", "chào buổi sáng"]
        is_casual = any(pattern in message_lower for pattern in casual_patterns)
        
        # Only detect teaching intent if we have strong indicators and it's not casual conversation
        detected = len(indicators) >= 1 and not is_casual and len(message) > 20  # Require substantial content
        confidence = min(len(indicators) * 0.4, 0.6) if detected else 0.1  # Lower confidence for fallback
        
        logger.info(f"Fallback Analysis - Indicators: {indicators}, Casual: {is_casual}, Length: {len(message)}, Detected: {detected}")
        
        return {
            "detected": detected,
            "confidence": confidence,
            "indicators": indicators,
            "reasoning": f"fallback_analysis (casual={is_casual}, indicators={len(indicators)})"
        }
    
    def _detect_knowledge_gap(self, message: str, search_results: str) -> Dict[str, Any]:
        """Analyze if there's a knowledge gap"""
        # Extract similarity from search results
        similarity = self._extract_similarity_from_search(search_results)
        
        if similarity < 0.35:
            gap_type = "NEW_KNOWLEDGE"
            significance = "HIGH"
        elif similarity < 0.70:
            gap_type = "PARTIAL_KNOWLEDGE"
            significance = "MEDIUM"
        else:
            gap_type = "EXISTING_KNOWLEDGE"
            significance = "LOW"
        
        return {
            "detected": similarity < 0.70,
            "type": gap_type,
            "significance": significance
        }
    
    def _assess_learning_value(self, message: str) -> Dict[str, Any]:
        """Assess the learning value of the content"""
        # Educational value indicators
        educational_indicators = ["example", "ví dụ", "method", "phương pháp", "principle", "nguyên tắc"]
        educational_value = sum(1 for indicator in educational_indicators if indicator in message.lower())
        
        # Practical value indicators
        practical_indicators = ["how to", "cách", "implement", "triển khai", "use", "sử dụng"]
        practical_value = sum(1 for indicator in practical_indicators if indicator in message.lower())
        
        # Uniqueness (length and detail)
        uniqueness = min(len(message) / 100, 1.0)  # Normalize by length
        
        return {
            "educational_value": min(educational_value * 0.3, 1.0),
            "practical_value": min(practical_value * 0.3, 1.0), 
            "uniqueness": uniqueness
        }
    
    def _assess_content_quality(self, message: str) -> Dict[str, Any]:
        """Assess the quality of the content"""
        # Clarity indicators
        clarity_score = 0.7  # Default reasonable clarity
        if len(message) > 50:  # Longer messages tend to be clearer
            clarity_score += 0.2
        if any(punct in message for punct in [".", "!", "?"]):  # Proper punctuation
            clarity_score += 0.1
        
        # Completeness indicators
        completeness_score = min(len(message) / 200, 1.0)  # Based on message length
        
        # Accuracy indicators (simple heuristics)
        accuracy_indicators = []
        if re.search(r'\d+', message):  # Contains numbers/data
            accuracy_indicators.append("contains_data")
        if any(word in message.lower() for word in ["research", "study", "proven", "nghiên cứu"]):
            accuracy_indicators.append("research_based")
        
        return {
            "clarity": min(clarity_score, 1.0),
            "completeness": completeness_score,
            "accuracy": accuracy_indicators
        }
    
    def _extract_similarity_from_search(self, search_results: str) -> float:
        """Extract similarity score from search results"""
        try:
            match = re.search(r'Best Similarity Score: (\d+\.?\d*)', search_results)
            if match:
                return float(match.group(1))
        except:
            pass
        return 0.0
    
    def _categorize_similarity(self, similarity: float) -> str:
        """Categorize similarity score"""
        if similarity >= 0.70:
            return "HIGH (Update existing)"
        elif similarity >= 0.35:
            return "MEDIUM (Human decision)"
        else:
            return "LOW (Create new)"
    
    def _generate_learning_recommendation(self, teaching_intent: Dict, knowledge_gap: Dict, 
                                        learning_value: Dict, content_quality: Dict, 
                                        similarity_score: float) -> Dict[str, Any]:
        """Generate overall learning recommendation"""
        
        # Calculate overall learning score
        score = 0.0
        reasoning_parts = []
        
        # Teaching intent contributes 30%
        if teaching_intent['detected']:
            score += 0.3 * teaching_intent['confidence']
            reasoning_parts.append("strong teaching intent detected")
        
        # Knowledge gap contributes 25%
        if knowledge_gap['detected']:
            gap_weight = {"HIGH": 0.25, "MEDIUM": 0.15, "LOW": 0.05}
            score += gap_weight.get(knowledge_gap['significance'], 0.05)
            reasoning_parts.append(f"{knowledge_gap['significance'].lower()} knowledge gap")
        
        # Learning value contributes 25%
        value_score = (learning_value['educational_value'] + learning_value['practical_value']) / 2
        score += 0.25 * value_score
        if value_score > 0.5:
            reasoning_parts.append("high educational/practical value")
        
        # Content quality contributes 20%
        quality_score = (content_quality['clarity'] + content_quality['completeness']) / 2
        score += 0.20 * quality_score
        if quality_score > 0.7:
            reasoning_parts.append("good content quality")
        
        # Determine action based on score and similarity
        if score >= 0.7 and similarity_score < 0.35:
            action = "LEARN_NEW"
            next_steps = "Proceed with saving new knowledge"
        elif score >= 0.7 and similarity_score >= 0.35:
            action = "REQUEST_HUMAN_DECISION"
            next_steps = "Ask human whether to update existing or create new knowledge"
        elif score >= 0.4:
            action = "MAYBE_LEARN"
            next_steps = "Consider learning with human confirmation"
        else:
            action = "SKIP_LEARNING"
            next_steps = "Continue conversation without learning"
        
        return {
            "action": action,
            "confidence": min(score, 1.0),
            "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "low learning indicators",
            "next_steps": next_steps
        }


class HumanLearningTool:
    """Tool for LLM to request human input on learning decisions"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown"):
        self.user_id = user_id
        self.org_id = org_id
        self.name = "human_learning"
    
    def request_learning_decision(self, decision_type: str, context: str, 
                                options: List[str], additional_info: str = "") -> str:
        """
        LLM calls this to ask human for learning decisions
        
        Args:
            decision_type: Type of decision needed
            context: Context about what the decision is for
            options: Available options for human to choose from
            additional_info: Any additional information to help decision
            
        Returns:
            Decision request ID and status
        """
        try:
            logger.info(f"HumanLearningTool: Requesting decision '{decision_type}' for user {self.user_id}")
            
            # Create unique decision ID
            decision_id = f"learning_decision_{uuid.uuid4().hex[:8]}"
            
            # Store decision request
            decision_request = {
                "id": decision_id,
                "type": decision_type,
                "context": context,
                "options": options,
                "additional_info": additional_info,
                "user_id": self.user_id,
                "org_id": self.org_id,
                "status": "PENDING",
                "created_at": datetime.now().isoformat(),
                "expires_at": self._calculate_expiry()
            }
            
            PENDING_LEARNING_DECISIONS[decision_id] = decision_request
            
            # Log decision request for debugging
            logger.info(f"HumanDecision Created - ID: {decision_id}, Type: {decision_type}, "
                       f"Options: {len(options)}, Context: '{context[:50]}...'")
            
            # Format response for LLM
            return f"""
HUMAN_DECISION_REQUESTED:
- Decision ID: {decision_id}
- Type: {decision_type}
- Status: WAITING_FOR_HUMAN_INPUT
- Expires: {decision_request['expires_at']}

FRONTEND_ACTION_REQUIRED:
The frontend should display a learning decision UI with:
- Context: {context}
- Options: {', '.join(options)}
- Additional Info: {additional_info}

NEXT_STEPS:
1. Frontend displays decision interface
2. Human makes selection
3. Frontend calls /api/learning/decision endpoint
4. System continues based on human choice

Note: This decision will expire in 5 minutes if not answered.
            """.strip()
            
        except Exception as e:
            logger.error(f"HumanLearningTool error: {e}")
            return f"ERROR: Failed to create learning decision request - {str(e)}"
    
    def check_decision_status(self, decision_id: str) -> str:
        """Check the status of a learning decision"""
        if decision_id not in PENDING_LEARNING_DECISIONS:
            return f"DECISION_NOT_FOUND: {decision_id}"
        
        decision = PENDING_LEARNING_DECISIONS[decision_id]
        
        if decision["status"] == "COMPLETED":
            return f"""
DECISION_COMPLETED:
- Decision ID: {decision_id}
- Human Choice: {decision.get('human_choice', 'unknown')}
- Completed At: {decision.get('completed_at', 'unknown')}
            """.strip()
        elif decision["status"] == "EXPIRED":
            return f"DECISION_EXPIRED: {decision_id}"
        else:
            return f"DECISION_PENDING: {decision_id} (waiting for human input)"
    
    def _calculate_expiry(self) -> str:
        """Calculate when the decision expires"""
        from datetime import timedelta
        expiry_time = datetime.now() + timedelta(minutes=5)
        return expiry_time.isoformat()
    
    def get_tool_description(self) -> Dict[str, Any]:
        """Get tool description for LLM function calling"""
        return {
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
        }


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
            logger.info(f"KnowledgePreviewTool: Previewing save for format '{save_format}'")
            
            # Generate different preview formats
            previews = {}
            
            if save_format in ["conversation", "all"]:
                previews["conversation"] = f"User: {user_message}\n\nAI: {ai_response}"
            
            if save_format in ["synthesis", "all"]:
                previews["synthesis"] = self._extract_synthesis_content(ai_response)
            
            if save_format in ["summary", "all"]:
                previews["summary"] = self._generate_summary(user_message, ai_response)
            
            # Calculate storage estimates
            total_chars = sum(len(content) for content in previews.values())
            estimated_tokens = total_chars // 4  # Rough estimate
            
            # Format preview
            preview_text = ""
            for format_type, content in previews.items():
                preview_text += f"\n=== {format_type.upper()} FORMAT ===\n{content}\n"
            
            return f"""
KNOWLEDGE SAVE PREVIEW:

CONTENT TO BE SAVED:
{preview_text}

METADATA:
- Save Format(s): {save_format}
- Total Characters: {total_chars}
- Estimated Tokens: {estimated_tokens}
- User ID: {self.user_id}
- Organization: {self.org_id}
- Categories: ["teaching_intent", "human_approved"]

STORAGE IMPACT:
- Vector Embeddings: {len(previews)} vectors will be created
- Search Impact: This content will be findable in future searches
- Update Impact: May affect similarity calculations for future content

NEXT_STEPS:
If you want to proceed with saving, call the knowledge_save tool.
If you want to modify the content first, call this preview tool again with changes.
            """.strip()
            
        except Exception as e:
            logger.error(f"KnowledgePreviewTool error: {e}")
            return f"ERROR: Failed to preview knowledge save - {str(e)}"
    
    def _extract_synthesis_content(self, ai_response: str) -> str:
        """Extract synthesis content from AI response"""
        # Look for structured sections
        synthesis_patterns = [
            r'<knowledge_synthesis>(.*?)</knowledge_synthesis>',
            r'<synthesis>(.*?)</synthesis>',
            r'SYNTHESIS:(.*?)(?=\n[A-Z]+:|$)',
        ]
        
        for pattern in synthesis_patterns:
            match = re.search(pattern, ai_response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: return first paragraph if no structured content
        paragraphs = ai_response.split('\n\n')
        if paragraphs:
            return paragraphs[0].strip()
        
        return ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
    
    def _generate_summary(self, user_message: str, ai_response: str) -> str:
        """Generate a summary of the knowledge"""
        # Extract key concepts
        user_concepts = self._extract_key_concepts(user_message)
        ai_concepts = self._extract_key_concepts(ai_response)
        
        # Combine unique concepts
        all_concepts = list(set(user_concepts + ai_concepts))
        
        return f"Summary: Discussion about {', '.join(all_concepts[:5])} covering key aspects shared by user and elaborated by AI."
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        # Filter common words and return most frequent
        common_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'same', 'right', 'used', 'take', 'three', 'want', 'different', 'new', 'good', 'need', 'way', 'well', 'without', 'most', 'these', 'come', 'might', 'every', 'since', 'many', 'back', 'great', 'year', 'years', 'such', 'important', 'because', 'some', 'people', 'system', 'example', 'information', 'using', 'process', 'approach'}
        
        filtered_words = [word for word in words if word not in common_words and len(word) > 4]
        
        # Return top 10 most frequent
        from collections import Counter
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(10)]
    
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
    """Tool for LLM to actually save knowledge (with human approval)"""
    
    def __init__(self, user_id: str = "unknown", org_id: str = "unknown"):
        self.user_id = user_id
        self.org_id = org_id
        self.name = "knowledge_save"
    
    async def save_knowledge(self, content: str, title: str = "", categories: List[str] = None, 
                      thread_id: str = None, decision_id: str = None) -> str:
        """
        LLM calls this to save knowledge (requires human approval)
        
        Args:
            content: The knowledge content to save
            title: Optional title for the knowledge
            categories: Categories to tag the knowledge with
            thread_id: Thread ID for conversation context
            decision_id: Reference to human decision (if applicable)
            
        Returns:
            Save status and result
        """
        try:
            logger.info(f"KnowledgeSaveTool: [STEP 1] Starting save for user {self.user_id}")
            
            # Validate human approval if decision_id provided
            if decision_id:
                logger.info(f"KnowledgeSaveTool: [STEP 2] Checking human approval for decision {decision_id}")
                approval_status = self._check_human_approval(decision_id)
                if not approval_status["approved"]:
                    return f"SAVE_REJECTED: {approval_status['reason']}"
            
            logger.info(f"KnowledgeSaveTool: [STEP 3] Preparing save parameters")
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
                    "saved_at": datetime.now().isoformat()
                }
            }
            
            logger.info(f"KnowledgeSaveTool: [STEP 4] Calling _perform_knowledge_save")
            # Perform the actual save
            save_result = await self._perform_knowledge_save(save_params)
            
            logger.info(f"KnowledgeSaveTool: [STEP 5] Save completed with result: {save_result}")
            # Log save attempt for debugging
            logger.info(f"KnowledgeSave Attempt - Success: {save_result['success']}, "
                       f"Title: '{title[:30]}...', Content Length: {len(content)}, "
                       f"Decision ID: {decision_id}")
            
            if save_result["success"]:
                return f"""
KNOWLEDGE_SAVED_SUCCESSFULLY:
- Knowledge ID: {save_result.get('knowledge_id', 'unknown')}
- Title: {save_params['title']}
- Categories: {', '.join(save_params['categories'])}
- Content Length: {len(content)} characters
- Vectors Created: {save_result.get('vectors_created', 1)}

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
            return {"approved": False, "reason": "Decision not found"}
        
        decision = PENDING_LEARNING_DECISIONS[decision_id]
        
        if decision["status"] != "COMPLETED":
            return {"approved": False, "reason": "Decision not completed"}
        
        human_choice = decision.get("human_choice", "")
        
        # Check if choice indicates approval
        approval_choices = ["save_new", "update_existing", "yes", "approve", "proceed"]
        approved = any(choice in human_choice.lower() for choice in approval_choices)
        
        return {"approved": approved, "reason": "Approved by human" if approved else "Human chose not to save"}
    
    async def _perform_knowledge_save(self, save_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual knowledge saving using the correct save_knowledge function like AVA does"""
        try:
            # Use the same save_knowledge function that AVA uses for Pinecone
            from pccontroller import save_knowledge
            
            logger.info(f"Calling pccontroller.save_knowledge for decision-based save")
            
            # Call save_knowledge with the same parameters AVA uses
            result = await save_knowledge(
                input=save_params["content"],
                user_id=save_params["user_id"],
                org_id=save_params["org_id"],
                title=save_params["title"],
                bank_name="conversation",
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
            "description": "Actually save knowledge (requires human approval)",
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
                        "description": "Reference to human decision",
                        "default": None
                    }
                },
                "required": ["content"]
            }
        }


class LearningToolsFactory:
    """Factory for creating and managing learning tools"""
    
    @staticmethod
    def create_learning_tools(user_id: str = "unknown", org_id: str = "unknown") -> List[Any]:
        """Create all learning tools for a user/org"""
        return [
            LearningSearchTool(user_id, org_id),
            LearningAnalysisTool(user_id, org_id),
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
                "description": "Actually save knowledge (requires human approval)",
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
                            "description": "Reference to human decision",
                            "default": None
                        }
                    },
                    "required": ["content"]
                }
            }
        ]


# Utility functions for decision management
def get_pending_decisions(user_id: str = None) -> Dict[str, Any]:
    """Get pending learning decisions for a user"""
    if user_id:
        return {k: v for k, v in PENDING_LEARNING_DECISIONS.items() if v.get("user_id") == user_id}
    return PENDING_LEARNING_DECISIONS.copy()


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
            
            # Create KnowledgeSaveTool instance and actually save the knowledge
            save_tool = KnowledgeSaveTool(user_id=user_id, org_id=org_id)
            
            # Prepare knowledge content
            knowledge_content = f"Human-approved teaching content:\n\nUser: {user_message}"
            knowledge_title = f"Knowledge from {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            knowledge_categories = ["teaching_intent", "human_approved", "interactive_learning"]
            
            # Call the actual save_knowledge function
            logger.info(f"Calling save_knowledge for decision {decision_id}")
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