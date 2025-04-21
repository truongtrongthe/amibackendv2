import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
import time
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

class ContactAnalyzer:
    """
    Analyzes contact profiles for sales signals and calculates sales readiness scores.
    Identifies opportunities and prioritizes contacts for follow-up.
    """
    
    def __init__(self):
        self.signal_keywords = {
            'interest': [
                'interested', 'looking for', 'considering', 'need', 'want', 'searching',
                'exploring', 'researching', 'evaluating', 'comparing', 'seeking'
            ],
            'urgency': [
                'urgent', 'immediate', 'asap', 'soon', 'quickly', 'deadline', 'running out',
                'limited time', 'end of month', 'end of quarter', 'by next week'
            ],
            'pain_points': [
                'problem', 'challenge', 'difficult', 'frustrated', 'struggling', 'issue',
                'unhappy', 'dissatisfied', 'concerned', 'worried', 'inefficient', 'costly'
            ],
            'budget': [
                'budget', 'investment', 'price', 'cost', 'affordable', 'funding',
                'financial', 'spending', 'expense', 'roi', 'return on investment'
            ],
            'decision': [
                'decide', 'decision', 'approve', 'approval', 'authority', 'responsible for',
                'in charge of', 'evaluate', 'sign off', 'green light', 'go ahead'
            ]
        }
        
        # Common job titles that indicate decision-making authority
        self.decision_maker_titles = [
            'ceo', 'cto', 'cio', 'cfo', 'chief', 'director', 'head', 'vp', 'vice president',
            'president', 'owner', 'founder', 'manager', 'executive', 'principal'
        ]
        
        self.analysis_table = "contact_analysis"
        
    def analyze_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a contact profile and return sales signals and a sales readiness score.
        
        Args:
            profile: The contact profile data dictionary
            
        Returns:
            Dict containing sales_signals, sales_readiness_score, and priority
        """
        signals = []
        scores = {
            'explicit_interest': 0,
            'urgency': 0,
            'decision_authority': 0,
            'need_alignment': 0,
            'engagement_level': 0
        }
        
        # Analyze profile summary
        if profile.get('profile_summary'):
            summary_signals = self._analyze_text(profile['profile_summary'])
            signals.extend([f"Profile Summary: {s}" for s in summary_signals])
            
            # Score explicit interest from summary
            if any(kw in profile['profile_summary'].lower() for kw in self.signal_keywords['interest']):
                scores['explicit_interest'] += 7
                
            # Score urgency from summary
            if any(kw in profile['profile_summary'].lower() for kw in self.signal_keywords['urgency']):
                scores['urgency'] += 5
                
            # Score need alignment from summary
            if any(kw in profile['profile_summary'].lower() for kw in self.signal_keywords['pain_points']):
                scores['need_alignment'] += 5
        
        # Analyze personality
        if profile.get('personality'):
            personality_signals = self._analyze_text(profile['personality'])
            signals.extend([f"Personality: {s}" for s in personality_signals])
        
        # Analyze hidden desires (high-value insights)
        if profile.get('hidden_desires'):
            desire_signals = self._analyze_text(profile['hidden_desires'])
            signals.extend([f"Hidden Desire: {s}" for s in desire_signals])
            
            # Score urgency from hidden desires
            if any(kw in profile['hidden_desires'].lower() for kw in self.signal_keywords['urgency']):
                scores['urgency'] += 5
                
            # Score need alignment from hidden desires
            if any(kw in profile['hidden_desires'].lower() for kw in self.signal_keywords['pain_points']):
                scores['need_alignment'] += 5
        
        # Analyze general info (often stored as JSON string)
        general_info = profile.get('general_info', '{}')
        if isinstance(general_info, str):
            try:
                general_info = json.loads(general_info)
            except (json.JSONDecodeError, TypeError):
                general_info = {}
        
        # Check for decision maker indicators in job title
        job_title = general_info.get('job_title', '').lower()
        if any(title in job_title for title in self.decision_maker_titles):
            signals.append(f"Decision Maker: Job title indicates authority ({job_title})")
            scores['decision_authority'] += 10
        
        # Analyze company size/industry if available
        if general_info.get('company'):
            signals.append(f"Company: {general_info.get('company')}")
            scores['need_alignment'] += 3
        
        if general_info.get('industry'):
            signals.append(f"Industry: {general_info.get('industry')}")
            scores['need_alignment'] += 2
            
        # Analyze best_goals (highest value signals)
        if profile.get('best_goals') and isinstance(profile['best_goals'], list):
            for goal in profile['best_goals']:
                if not isinstance(goal, dict):
                    continue
                    
                goal_text = goal.get('goal_name', '') or goal.get('goal', '')
                goal_description = goal.get('description', '')
                if goal_description:
                    goal_text = f"{goal_text} - {goal_description}"
                    
                goal_importance = goal.get('importance', '').lower()
                goal_deadline_str = goal.get('deadline', '')
                
                # Add as a signal
                goal_signal = f"Goal: {goal_text}"
                if goal_importance:
                    goal_signal += f" (Importance: {goal_importance})"
                if goal_deadline_str:
                    goal_signal += f" (Deadline: {goal_deadline_str})"
                signals.append(goal_signal)
                
                # Score explicit interest from goals
                if any(kw in goal_text.lower() for kw in self.signal_keywords['interest']):
                    scores['explicit_interest'] += 8
                    
                # Score explicit interest from purchase-intent keywords
                if any(kw in goal_text.lower() for kw in ['buy', 'purchase', 'acquire', 'get', 'implement']):
                    scores['explicit_interest'] += 15
                
                # Score urgency from goal deadline
                if goal_deadline_str:
                    try:
                        deadline = datetime.strptime(goal_deadline_str, '%Y-%m-%d')
                        today = datetime.now()
                        days_until_deadline = (deadline - today).days
                        
                        if days_until_deadline <= 30:  # Within a month
                            scores['urgency'] += 15
                        elif days_until_deadline <= 90:  # Within 3 months
                            scores['urgency'] += 10
                        elif days_until_deadline <= 180:  # Within 6 months
                            scores['urgency'] += 5
                    except ValueError:
                        # If date parsing fails, just continue
                        pass
                
                # Score by importance
                if goal_importance == 'high':
                    scores['explicit_interest'] += 7
                    scores['urgency'] += 5
                elif goal_importance == 'medium':
                    scores['explicit_interest'] += 4
                    scores['urgency'] += 3
                
                # Score decision authority from budget-related goals
                if any(kw in goal_text.lower() for kw in self.signal_keywords['budget']):
                    scores['decision_authority'] += 10
                
                # Score need alignment from pain-point goals
                if any(kw in goal_text.lower() for kw in self.signal_keywords['pain_points']):
                    scores['need_alignment'] += 5
        
        # Check for LinkedIn URL as a signal of professional engagement
        if profile.get('linkedin_url'):
            signals.append(f"LinkedIn Profile: Available")
            scores['engagement_level'] += 3
        
        # Check for social media presence
        if profile.get('social_media_urls') and isinstance(profile['social_media_urls'], list):
            platform_count = len(profile['social_media_urls'])
            if platform_count > 0:
                signals.append(f"Social Media: Present on {platform_count} platforms")
                scores['engagement_level'] += min(platform_count, 5)  # Cap at 5 points
        
        # Calculate total score (weighted)
        total_score = (
            scores['explicit_interest'] * 0.3 +
            scores['urgency'] * 0.25 +
            scores['decision_authority'] * 0.2 +
            scores['need_alignment'] * 0.15 +
            scores['engagement_level'] * 0.1
        )
        
        # Normalize to 0-100 scale
        normalized_score = min(round(total_score * 2), 100)
        
        # Determine priority level
        priority = self._determine_priority(normalized_score)
        
        return {
            'sales_signals': signals,
            'score_breakdown': scores,
            'sales_readiness_score': normalized_score,
            'priority': priority,
            'analyzed_at': datetime.utcnow().isoformat()
        }
    
    def _analyze_text(self, text: str) -> List[str]:
        """
        Analyze text for sales signals based on keyword matching.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of identified signals as strings
        """
        signals = []
        text_lower = text.lower()
        
        # Check for interest signals
        for keyword in self.signal_keywords['interest']:
            if keyword in text_lower:
                signals.append(f"Shows interest ({keyword})")
                break
        
        # Check for urgency signals
        for keyword in self.signal_keywords['urgency']:
            if keyword in text_lower:
                signals.append(f"Indicates urgency ({keyword})")
                break
        
        # Check for pain points
        for keyword in self.signal_keywords['pain_points']:
            if keyword in text_lower:
                signals.append(f"Mentions pain point ({keyword})")
                break
        
        # Check for budget references
        for keyword in self.signal_keywords['budget']:
            if keyword in text_lower:
                signals.append(f"References budget ({keyword})")
                break
        
        # Check for decision-making capacity
        for keyword in self.signal_keywords['decision']:
            if keyword in text_lower:
                signals.append(f"Indicates decision authority ({keyword})")
                break
        
        return signals
    
    def _determine_priority(self, score: int) -> str:
        """
        Determine priority level based on sales readiness score.
        
        Args:
            score: The sales readiness score (0-100)
            
        Returns:
            Priority level as string
        """
        if score >= 80:
            return "High (Hot Lead)"
        elif score >= 60:
            return "Medium (Warm Lead)"
        elif score >= 40:
            return "Low (Nurture)"
        else:
            return "Very Low (Unqualified)"
    
    def bulk_analyze_contacts(self, contacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple contacts and return results with priority sorting.
        
        Args:
            contacts: List of contacts with their profiles
            
        Returns:
            List of contacts with added analysis and sorted by priority
        """
        results = []
        
        logger.info(f"Bulk analyzing {len(contacts)} contacts")
        
        for contact in contacts:
            profile = contact.get('profiles')
            if not profile:
                logger.debug(f"Skipping contact {contact.get('id')}: no profile found")
                continue
            
            logger.info(f"Analyzing profile for contact {contact.get('id')} - profile: {profile.get('id')}")
            
            try:
                # Analyze the profile
                analysis = self.analyze_profile(profile)
                
                # Add analysis to contact
                contact_with_analysis = {
                    **contact,
                    'sales_analysis': analysis
                }
                
                results.append(contact_with_analysis)
                logger.info(f"Successfully analyzed contact {contact.get('id')} with score {analysis.get('sales_readiness_score', 0)}")
            except Exception as e:
                logger.error(f"Error analyzing contact {contact.get('id')}: {str(e)}")
        
        logger.info(f"Successfully analyzed {len(results)}/{len(contacts)} contacts")
        
        # Sort by priority (highest score first)
        results.sort(key=lambda x: x['sales_analysis']['sales_readiness_score'], reverse=True)
        
        return results
    
    def generate_weekly_leads_report(self, contacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a weekly report of lead priorities and opportunities.
        
        Args:
            contacts: List of contacts with their profiles
            
        Returns:
            Report dictionary with lead statistics and top opportunities
        """
        analyzed_contacts = self.bulk_analyze_contacts(contacts)
        
        # Initialize report structure
        report = {
            'report_date': datetime.utcnow().isoformat(),
            'total_contacts': len(contacts),
            'total_with_profiles': len(analyzed_contacts),
            'lead_breakdown': {
                'hot_leads': 0,
                'warm_leads': 0,
                'nurture_leads': 0,
                'unqualified_leads': 0
            },
            'top_opportunities': [],
            'urgent_opportunities': []  # Opportunities with high urgency score
        }
        
        # Categorize leads
        for contact in analyzed_contacts:
            priority = contact['sales_analysis']['priority']
            
            if "High" in priority:
                report['lead_breakdown']['hot_leads'] += 1
                # Add to top opportunities if we have less than 10
                if len(report['top_opportunities']) < 10:
                    report['top_opportunities'].append({
                        'contact_id': contact.get('id'),
                        'name': f"{contact.get('first_name', '')} {contact.get('last_name', '')}",
                        'score': contact['sales_analysis']['sales_readiness_score'],
                        'top_signals': contact['sales_analysis']['sales_signals'][:3]  # Top 3 signals
                    })
            elif "Medium" in priority:
                report['lead_breakdown']['warm_leads'] += 1
            elif "Low" in priority:
                report['lead_breakdown']['nurture_leads'] += 1
            else:
                report['lead_breakdown']['unqualified_leads'] += 1
            
            # Check for urgent opportunities (high urgency score)
            if contact['sales_analysis']['score_breakdown']['urgency'] >= 15:
                report['urgent_opportunities'].append({
                    'contact_id': contact.get('id'),
                    'name': f"{contact.get('first_name', '')} {contact.get('last_name', '')}",
                    'urgency_score': contact['sales_analysis']['score_breakdown']['urgency'],
                    'urgency_signals': [s for s in contact['sales_analysis']['sales_signals'] 
                                      if "urgency" in s.lower() or "deadline" in s.lower()]
                })
        
        # Ensure urgent opportunities are unique and limited to top 5
        seen_ids = set()
        unique_urgent = []
        for opp in report['urgent_opportunities']:
            if opp['contact_id'] not in seen_ids:
                seen_ids.add(opp['contact_id'])
                unique_urgent.append(opp)
                if len(unique_urgent) >= 5:
                    break
        
        report['urgent_opportunities'] = unique_urgent
        
        return report
    
    def track_signal_changes(
        self, 
        current_analysis: Dict[str, Any], 
        previous_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Track changes in sales signals between analyses.
        
        Args:
            current_analysis: The current sales signal analysis
            previous_analysis: The previous sales signal analysis (if available)
            
        Returns:
            Dictionary with signal changes and trends
        """
        if not previous_analysis:
            return {
                'is_first_analysis': True,
                'signal_changes': [],
                'score_change': 0,
                'trend': 'initial'
            }
        
        # Calculate score change
        current_score = current_analysis.get('sales_readiness_score', 0)
        previous_score = previous_analysis.get('sales_readiness_score', 0)
        score_change = current_score - previous_score
        
        # Determine trend
        if score_change > 5:
            trend = 'increasing'
        elif score_change < -5:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Identify new and disappeared signals
        current_signals = set(current_analysis.get('sales_signals', []))
        previous_signals = set(previous_analysis.get('sales_signals', []))
        
        new_signals = current_signals - previous_signals
        disappeared_signals = previous_signals - current_signals
        
        # Format changes
        signal_changes = []
        
        for signal in new_signals:
            signal_changes.append({
                'signal': signal,
                'change_type': 'new',
                'change_date': datetime.utcnow().isoformat()
            })
        
        for signal in disappeared_signals:
            signal_changes.append({
                'signal': signal,
                'change_type': 'disappeared',
                'change_date': datetime.utcnow().isoformat()
            })
        
        return {
            'is_first_analysis': False,
            'signal_changes': signal_changes,
            'score_change': score_change,
            'trend': trend
        }
    
    def store_analysis(self, contact_id: int, organization_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store the analysis results in the database
        
        Args:
            contact_id: The ID of the contact
            organization_id: The organization ID
            analysis: The analysis results from analyze_profile
            
        Returns:
            The stored analysis record
        """
        try:
            # Format the data for insertion
            analysis_data = {
                "contact_id": contact_id,
                "organization_id": organization_id,
                "sales_signals": json.dumps(analysis.get('sales_signals', [])),
                "score_breakdown": json.dumps(analysis.get('score_breakdown', {})),
                "sales_readiness_score": analysis.get('sales_readiness_score', 0),
                "priority": analysis.get('priority', "Very Low (Unqualified)"),
                "analyzed_at": analysis.get('analyzed_at', datetime.utcnow().isoformat())
            }
            
            # Insert into database
            response = supabase.table(self.analysis_table).insert(analysis_data).execute()
            
            if response.data:
                logger.info(f"Stored analysis for contact {contact_id} with score {analysis.get('sales_readiness_score', 0)}")
                return response.data[0]
            else:
                logger.error(f"Failed to store analysis for contact {contact_id}")
                return None
        except Exception as e:
            logger.error(f"Error storing analysis: {str(e)}")
            return None
    
    def get_analysis_history(self, contact_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical analysis for a contact
        
        Args:
            contact_id: The ID of the contact
            limit: Maximum number of records to return
            
        Returns:
            List of historical analysis records
        """
        try:
            response = supabase.table(self.analysis_table)\
                .select("*")\
                .eq("contact_id", contact_id)\
                .order("analyzed_at", desc=True)\
                .limit(limit)\
                .execute()
            
            history = []
            for record in response.data:
                # Convert JSON strings back to objects
                if 'sales_signals' in record and isinstance(record['sales_signals'], str):
                    try:
                        record['sales_signals'] = json.loads(record['sales_signals'])
                    except json.JSONDecodeError:
                        record['sales_signals'] = []
                        
                if 'score_breakdown' in record and isinstance(record['score_breakdown'], str):
                    try:
                        record['score_breakdown'] = json.loads(record['score_breakdown'])
                    except json.JSONDecodeError:
                        record['score_breakdown'] = {}
                        
                history.append(record)
                
            return history
        except Exception as e:
            logger.error(f"Error retrieving analysis history: {str(e)}")
            return []
    
    def get_latest_analysis(self, contact_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the most recent analysis for a contact
        
        Args:
            contact_id: The ID of the contact
            
        Returns:
            The most recent analysis record or None
        """
        history = self.get_analysis_history(contact_id, limit=1)
        return history[0] if history else None
    
    def analyze_and_store(self, contact_id: int, organization_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a contact profile and store the results
        
        Args:
            contact_id: The ID of the contact
            organization_id: The organization ID
            profile: The contact profile data
            
        Returns:
            The analysis results and comparison with previous analysis
        """
        # Get the current analysis
        analysis = self.analyze_profile(profile)
        
        # Store the analysis
        stored_analysis = self.store_analysis(contact_id, organization_id, analysis)
        
        # Get the previous analysis
        previous_analyses = self.get_analysis_history(contact_id, limit=2)
        previous_analysis = previous_analyses[1] if len(previous_analyses) > 1 else None
        
        # Track changes from previous analysis
        changes = self.track_signal_changes(analysis, previous_analysis)
        
        return {
            "analysis": analysis,
            "changes": changes,
            "stored": stored_analysis is not None
        }
    
    def get_hot_leads(self, organization_id: str, min_score: int = 70, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get hot leads for an organization based on sales readiness score
        
        Args:
            organization_id: The organization ID
            min_score: Minimum score to consider a lead "hot"
            limit: Maximum number of leads to return
            
        Returns:
            List of hot leads with their latest analysis
        """
        try:
            # Get the latest analysis for each contact in the organization
            # This requires a more complex query to get the latest analysis per contact
            query = f"""
            WITH latest_analyses AS (
                SELECT DISTINCT ON (contact_id) 
                    id, contact_id, sales_readiness_score, priority, analyzed_at
                FROM {self.analysis_table}
                WHERE organization_id = '{organization_id}'
                ORDER BY contact_id, analyzed_at DESC
            )
            SELECT * FROM latest_analyses
            WHERE sales_readiness_score >= {min_score}
            ORDER BY sales_readiness_score DESC
            LIMIT {limit}
            """
            
            response = supabase.rpc('exec_sql', { 'query': query }).execute()
            
            if not response.data:
                return []
                
            # Get contact details for these hot leads
            hot_leads = []
            for analysis in response.data:
                contact_response = supabase.table("contacts")\
                    .select("*, profiles(*)")\
                    .eq("id", analysis['contact_id'])\
                    .execute()
                    
                if contact_response.data:
                    contact = contact_response.data[0]
                    hot_leads.append({
                        "contact": contact,
                        "latest_analysis": analysis
                    })
                    
            return hot_leads
        except Exception as e:
            logger.error(f"Error getting hot leads: {str(e)}")
            return []
            
    def batch_analyze_contacts(self, organization_id: Optional[str] = None, 
                              min_contacts: int = 10, 
                              max_contacts: int = 100,
                              batch_size: int = 10,
                              delay_seconds: float = 0.5) -> Dict:
        """
        Run batch analysis on contact profiles to detect sales signals and 
        assign priority levels.
        
        Args:
            organization_id: Optional organization ID to filter contacts
            min_contacts: Minimum number of contacts to process
            max_contacts: Maximum number of contacts to process
            batch_size: Number of contacts to process in each batch
            delay_seconds: Delay between batches to avoid rate limiting
            
        Returns:
            Dict with analysis results summary
        """
        from contact import ContactManager
        contact_manager = ContactManager()
        
        # Get contacts with profiles for the organization
        logger.info(f"Fetching contacts for organization_id: {organization_id or 'all'}")
        contacts = contact_manager.get_contacts(organization_id)
        
        # Filter contacts that have profiles
        contacts_with_profiles = [c for c in contacts if c.get("profiles")]
        
        if not contacts_with_profiles:
            logger.warning(f"No contacts with profiles found for organization_id: {organization_id or 'all'}")
            return {
                "success": False,
                "message": "No contacts with profiles found",
                "processed": 0,
                "total": 0
            }
        
        # Limit to max_contacts
        contacts_to_process = contacts_with_profiles[:max_contacts]
        total_contacts = len(contacts_to_process)
        
        if total_contacts < min_contacts:
            logger.warning(f"Found only {total_contacts} contacts with profiles, which is below minimum threshold of {min_contacts}")
            return {
                "success": False,
                "message": f"Only {total_contacts} contacts with profiles found, which is below minimum threshold of {min_contacts}",
                "processed": 0,
                "total": total_contacts
            }
        
        logger.info(f"Starting batch analysis of {total_contacts} contacts for organization_id: {organization_id or 'all'}")
        
        # Process contacts in batches
        processed_count = 0
        success_count = 0
        error_count = 0
        
        for i in range(0, total_contacts, batch_size):
            batch = contacts_to_process[i:i+batch_size]
            batch_number = i // batch_size + 1
            batch_size_actual = len(batch)
            
            logger.info(f"Processing batch {batch_number}/{(total_contacts + batch_size - 1) // batch_size} with {batch_size_actual} contacts")
            
            for contact in batch:
                contact_id = contact.get("id")
                profile = contact.get("profiles")
                
                if not profile:
                    logger.warning(f"Skipping contact {contact_id}: profile data missing")
                    continue
                    
                try:
                    # Get organization_id from contact if not provided
                    org_id = organization_id or contact.get("organization_id")
                    if not org_id:
                        logger.warning(f"Skipping contact {contact_id}: organization_id missing")
                        continue
                    
                    # Analyze and store results
                    result = self.analyze_and_store(contact_id, org_id, profile)
                    
                    if result.get("stored"):
                        success_count += 1
                        logger.info(f"Successfully analyzed contact {contact_id} with score {result['analysis'].get('sales_readiness_score', 0)}")
                    else:
                        error_count += 1
                        logger.error(f"Failed to store analysis for contact {contact_id}")
                    
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error analyzing contact {contact_id}: {str(e)}")
            
            # Add delay between batches
            if batch_number * batch_size < total_contacts:
                logger.info(f"Waiting {delay_seconds} seconds before processing next batch")
                time.sleep(delay_seconds)
        
        # Generate summary
        summary = {
            "success": True,
            "message": f"Processed {processed_count} contacts with {success_count} successes and {error_count} errors",
            "processed": processed_count,
            "success_count": success_count,
            "error_count": error_count,
            "total": total_contacts
        }
        
        # Create weekly report if requested
        if processed_count > 0:
            try:
                report = self.generate_weekly_leads_report(contacts_to_process)
                summary["report"] = report
                
                logger.info(f"Generated sales report: {report['lead_breakdown']['hot_leads']} hot leads, " +
                          f"{report['lead_breakdown']['warm_leads']} warm leads, " +
                          f"{report['lead_breakdown']['nurture_leads']} nurture leads")
                          
            except Exception as e:
                logger.error(f"Error generating weekly report: {str(e)}")
            
            # Refresh contact tags after analysis is complete
            try:
                logger.info("Refreshing contact tags based on latest analysis")
                # This will trigger _enrich_contacts_with_tags to update all tags
                contact_manager.get_contacts(organization_id)
                logger.info("Contact tags refreshed successfully")
            except Exception as e:
                logger.error(f"Error refreshing contact tags: {str(e)}")
        
        return summary 