"""
Business Logic Tool - Domain-specific business operations
Handles sales analysis, reporting, and other business intelligence tasks
"""

import logging
from typing import Dict, Any, List
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Configure business logic specific logging
business_logger = logging.getLogger("business_logic")
business_logger.setLevel(logging.INFO)
if not business_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('📊 [BUSINESS] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    business_logger.addHandler(handler)


class BusinessLogicTool:
    """Tool for business-specific operations and analysis"""
    
    def __init__(self):
        self.name = "business_logic_tool"
    
    def sale_summarize(self, data: str, instructions: str) -> str:
        """
        Summarize sales data based on instructions
        
        Args:
            data: Raw sales data from file
            instructions: Instructions on how to summarize (from knowledge base)
            
        Returns:
            Formatted sales summary
        """
        business_logger.info(f"SALE_SUMMARIZE - Starting analysis")
        business_logger.info(f"SALE_SUMMARIZE - Data length: {len(data)} chars")
        business_logger.info(f"SALE_SUMMARIZE - Instructions: {instructions[:100]}{'...' if len(instructions) > 100 else ''}")
        
        try:
            summary_parts = []
            summary_parts.append("📊 QUARTERLY SALES REPORT")
            summary_parts.append("=" * 50)
            summary_parts.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_parts.append("")
            
            # Parse instructions for specific requirements
            instructions_lower = instructions.lower()
            business_logger.info(f"SALE_SUMMARIZE - Parsed instructions, analyzing requirements")
            
            if "revenue" in instructions_lower or "total" in instructions_lower:
                revenue_analysis = self._analyze_revenue(data)
                summary_parts.append("💰 REVENUE ANALYSIS:")
                summary_parts.append("-" * 20)
                summary_parts.extend(revenue_analysis)
                summary_parts.append("")
            
            if "trends" in instructions_lower or "growth" in instructions_lower:
                trends_analysis = self._analyze_trends(data) 
                summary_parts.append("📈 TRENDS ANALYSIS:")
                summary_parts.append("-" * 20)
                summary_parts.extend(trends_analysis)
                summary_parts.append("")
            
            if "top products" in instructions_lower or "best" in instructions_lower:
                products_analysis = self._analyze_top_products(data)
                summary_parts.append("🏆 TOP PERFORMING PRODUCTS:")
                summary_parts.append("-" * 20)
                summary_parts.extend(products_analysis)
                summary_parts.append("")
            
            if "regions" in instructions_lower or "geography" in instructions_lower:
                regions_analysis = self._analyze_regions(data)
                summary_parts.append("🌍 REGIONAL PERFORMANCE:")
                summary_parts.append("-" * 20)
                summary_parts.extend(regions_analysis)
                summary_parts.append("")
            
            # Add raw data summary
            summary_parts.append("📋 DATA SUMMARY:")
            summary_parts.append("-" * 20)
            summary_parts.append(f"• Data source analyzed: {len(data):,} characters")
            summary_parts.append(f"• Processing instructions: {instructions}")
            summary_parts.append(f"• Analysis completed: {datetime.now().strftime('%H:%M:%S')}")
            
            final_summary = "\n".join(summary_parts)
            business_logger.info(f"SALE_SUMMARIZE - Generated summary report ({len(final_summary)} chars)")
            business_logger.info(f"SALE_SUMMARIZE - Analysis completed successfully")
            
            return final_summary
            
        except Exception as e:
            business_logger.error(f"SALE_SUMMARIZE - Error: {str(e)}")
            return f"Error creating sales summary: {str(e)}"
    
    def _analyze_revenue(self, data: str) -> List[str]:
        """Extract revenue information from sales data"""
        results = []
        
        # Look for currency patterns (basic regex matching)
        currency_pattern = r'[\$€£¥]\s*[\d,]+\.?\d*'
        amounts = re.findall(currency_pattern, data)
        
        if amounts:
            results.append(f"• Found {len(amounts)} revenue entries")
            results.append(f"• Sample amounts: {', '.join(amounts[:5])}")
        else:
            results.append("• No clear revenue patterns detected in data")
        
        # Look for total/revenue keywords
        revenue_lines = [line.strip() for line in data.split('\n') 
                        if any(keyword in line.lower() for keyword in ['total', 'revenue', 'sales', 'income'])]
        
        if revenue_lines:
            results.append("• Key revenue mentions:")
            for line in revenue_lines[:3]:  # Show first 3 matches
                results.append(f"  - {line[:80]}{'...' if len(line) > 80 else ''}")
        
        return results
    
    def _analyze_trends(self, data: str) -> List[str]:
        """Analyze trends in the sales data"""
        results = []
        
        # Look for percentage changes
        percentage_pattern = r'[\+\-]?\d+\.?\d*%'
        percentages = re.findall(percentage_pattern, data)
        
        if percentages:
            results.append(f"• Found {len(percentages)} percentage changes")
            positive_changes = [p for p in percentages if p.startswith('+') or (not p.startswith('-') and float(p.replace('%', '')) > 0)]
            negative_changes = [p for p in percentages if p.startswith('-')]
            
            if positive_changes:
                results.append(f"• Positive trends: {', '.join(positive_changes[:3])}")
            if negative_changes:
                results.append(f"• Areas of concern: {', '.join(negative_changes[:3])}")
        
        # Look for time-based data
        month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
        months = re.findall(month_pattern, data, re.IGNORECASE)
        
        if months:
            unique_months = list(set([m.lower() for m in months]))
            results.append(f"• Time periods covered: {', '.join(unique_months[:6])}")
        
        return results or ["• No clear trend patterns detected in data"]
    
    def _analyze_top_products(self, data: str) -> List[str]:
        """Identify top performing products from sales data"""
        results = []
        
        # Look for product patterns (basic heuristics)
        product_lines = []
        for line in data.split('\n'):
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['product', 'item', 'sku', 'model']):
                product_lines.append(line)
        
        if product_lines:
            results.append(f"• Found {len(product_lines)} product-related entries")
            results.append("• Sample products mentioned:")
            for line in product_lines[:5]:  # Show first 5
                results.append(f"  - {line[:60]}{'...' if len(line) > 60 else ''}")
        else:
            # Look for table-like data that might contain products
            table_lines = [line for line in data.split('\n') if '|' in line and len(line.split('|')) > 2]
            if table_lines:
                results.append(f"• Found {len(table_lines)} table rows (may contain product data)")
                results.append("• Sample table entries:")
                for line in table_lines[:3]:
                    results.append(f"  - {line.strip()}")
            else:
                results.append("• No clear product patterns detected in data")
        
        return results
    
    def _analyze_regions(self, data: str) -> List[str]:
        """Analyze regional performance from sales data"""
        results = []
        
        # Common region/location keywords
        region_keywords = ['north', 'south', 'east', 'west', 'region', 'territory', 'area', 'zone', 
                          'usa', 'europe', 'asia', 'america', 'pacific', 'atlantic', 'central']
        
        region_lines = []
        for line in data.split('\n'):
            line = line.strip().lower()
            if any(keyword in line for keyword in region_keywords):
                region_lines.append(line)
        
        if region_lines:
            results.append(f"• Found {len(region_lines)} region-related entries")
            results.append("• Regional mentions:")
            for line in region_lines[:4]:  # Show first 4
                results.append(f"  - {line[:70]}{'...' if len(line) > 70 else ''}")
        else:
            results.append("• No clear regional patterns detected in data")
        
        return results
    
    def generate_executive_summary(self, data: str, key_metrics: Dict[str, Any]) -> str:
        """
        Generate executive summary from sales data and metrics
        
        Args:
            data: Raw sales data
            key_metrics: Dictionary of key performance metrics
            
        Returns:
            Executive summary text
        """
        try:
            summary_parts = []
            summary_parts.append("🎯 EXECUTIVE SUMMARY")
            summary_parts.append("=" * 30)
            summary_parts.append("")
            
            # Key highlights
            if key_metrics:
                summary_parts.append("📈 KEY HIGHLIGHTS:")
                for metric, value in key_metrics.items():
                    summary_parts.append(f"• {metric}: {value}")
                summary_parts.append("")
            
            # Data insights
            word_count = len(data.split())
            summary_parts.append("🔍 ANALYSIS SCOPE:")
            summary_parts.append(f"• Data analyzed: {len(data):,} characters, {word_count:,} words")
            summary_parts.append(f"• Report generated: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating executive summary: {str(e)}"
    
    def calculate_growth_metrics(self, current_data: str, previous_data: str = None) -> str:
        """
        Calculate growth metrics comparing current vs previous period
        
        Args:
            current_data: Current period sales data
            previous_data: Previous period sales data (optional)
            
        Returns:
            Growth metrics analysis
        """
        try:
            metrics_parts = []
            metrics_parts.append("📊 GROWTH METRICS")
            metrics_parts.append("=" * 25)
            metrics_parts.append("")
            
            if previous_data:
                # Compare data sizes as a basic metric
                current_size = len(current_data)
                previous_size = len(previous_data)
                growth_rate = ((current_size - previous_size) / previous_size) * 100
                
                metrics_parts.append("📈 DATA VOLUME COMPARISON:")
                metrics_parts.append(f"• Current period: {current_size:,} characters")
                metrics_parts.append(f"• Previous period: {previous_size:,} characters")
                metrics_parts.append(f"• Data growth: {growth_rate:+.1f}%")
            else:
                metrics_parts.append("⚠️ Previous period data not provided")
                metrics_parts.append("• Cannot calculate comparative metrics")
                metrics_parts.append(f"• Current period analyzed: {len(current_data):,} characters")
            
            metrics_parts.append("")
            metrics_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(metrics_parts)
            
        except Exception as e:
            return f"Error calculating growth metrics: {str(e)}"
    
    def get_tool_description(self):
        """Return tool descriptions for LLM function calling"""
        return [
            {
                "name": "sale_summarize",
                "description": "CRITICAL: Analyze and summarize business documents, sales reports, and financial data. Use this for comprehensive business analysis and reporting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The business data, sales report, or document content to analyze"
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Specific analysis instructions or focus areas (e.g., 'focus on revenue trends', 'analyze regional performance')"
                        }
                    },
                    "required": ["data", "instructions"]
                }
            },
            {
                "name": "generate_executive_summary",
                "description": "Generate executive summary from business data and key metrics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "Raw business data to summarize"
                        },
                        "key_metrics": {
                            "type": "object",
                            "description": "Dictionary of key performance metrics to highlight"
                        }
                    },
                    "required": ["data", "key_metrics"]
                }
            },
            {
                "name": "calculate_growth_metrics",
                "description": "Calculate growth metrics comparing current vs previous period data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_data": {
                            "type": "string",
                            "description": "Current period business data"
                        },
                        "previous_data": {
                            "type": "string",
                            "description": "Previous period business data (optional)"
                        }
                    },
                    "required": ["current_data"]
                }
            }
        ] 