"""
Insights Tool - Provides meaningful analytics and insights from data
"""

from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
import json
from datetime import datetime
import statistics


class InsightsTool(BaseTool):
    """Tool for generating meaningful insights from data analysis"""
    
    name: str = "insights_analyzer"
    description: str = """Analyze data and generate meaningful insights. 
    Input should be a JSON string with:
    - "data": array of numbers or objects to analyze
    - "type": analysis type (statistical, trend, comparison, summary)
    - "question": optional specific question to answer about the data
    
    Returns comprehensive insights including patterns, anomalies, and recommendations."""
    
    def __init__(self):
        super().__init__()
    
    def _run(self, input_str: str) -> str:
        """Run data analysis and generate insights"""
        try:
            params = json.loads(input_str)
            data = params.get("data", [])
            analysis_type = params.get("type", "summary")
            question = params.get("question", "")
            
            if not data:
                return json.dumps({"error": "No data provided for analysis"})
            
            insights = {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type,
                "data_points": len(data) if isinstance(data, list) else 1
            }
            
            if analysis_type == "statistical":
                insights.update(self._statistical_analysis(data))
            elif analysis_type == "trend":
                insights.update(self._trend_analysis(data))
            elif analysis_type == "comparison":
                insights.update(self._comparison_analysis(data))
            else:
                insights.update(self._summary_analysis(data))
            
            # Add recommendations
            insights["recommendations"] = self._generate_recommendations(insights)
            
            return json.dumps(insights, indent=2)
            
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid JSON input",
                "hint": "Provide data as: {\"data\": [...], \"type\": \"statistical\"}"
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _statistical_analysis(self, data: List) -> Dict[str, Any]:
        """Perform statistical analysis on numerical data"""
        try:
            # Extract numerical values
            numbers = self._extract_numbers(data)
            
            if not numbers:
                return {"error": "No numerical data found for statistical analysis"}
            
            n = len(numbers)
            mean_val = statistics.mean(numbers)
            
            result = {
                "statistics": {
                    "count": n,
                    "sum": sum(numbers),
                    "mean": round(mean_val, 4),
                    "median": statistics.median(numbers),
                    "min": min(numbers),
                    "max": max(numbers),
                    "range": max(numbers) - min(numbers)
                }
            }
            
            if n > 1:
                result["statistics"]["std_dev"] = round(statistics.stdev(numbers), 4)
                result["statistics"]["variance"] = round(statistics.variance(numbers), 4)
            
            # Detect outliers using IQR method
            if n >= 4:
                sorted_nums = sorted(numbers)
                q1 = statistics.median(sorted_nums[:n//2])
                q3 = statistics.median(sorted_nums[n//2 + n%2:])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [x for x in numbers if x < lower_bound or x > upper_bound]
                result["outliers"] = {
                    "count": len(outliers),
                    "values": outliers[:10],  # Limit to first 10
                    "bounds": {"lower": round(lower_bound, 4), "upper": round(upper_bound, 4)}
                }
            
            # Distribution insights
            result["distribution"] = {
                "skewness": "right-skewed" if mean_val > statistics.median(numbers) else "left-skewed" if mean_val < statistics.median(numbers) else "symmetric"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def _trend_analysis(self, data: List) -> Dict[str, Any]:
        """Analyze trends in sequential data"""
        try:
            numbers = self._extract_numbers(data)
            
            if len(numbers) < 2:
                return {"error": "Need at least 2 data points for trend analysis"}
            
            # Calculate trend direction
            changes = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            positive_changes = sum(1 for c in changes if c > 0)
            negative_changes = sum(1 for c in changes if c < 0)
            
            overall_change = numbers[-1] - numbers[0]
            percent_change = (overall_change / numbers[0] * 100) if numbers[0] != 0 else 0
            
            trend_direction = "increasing" if positive_changes > negative_changes else "decreasing" if negative_changes > positive_changes else "stable"
            
            result = {
                "trend": {
                    "direction": trend_direction,
                    "overall_change": round(overall_change, 4),
                    "percent_change": round(percent_change, 2),
                    "positive_periods": positive_changes,
                    "negative_periods": negative_changes,
                    "average_change": round(statistics.mean(changes), 4)
                }
            }
            
            # Momentum (recent vs historical)
            if len(numbers) >= 4:
                recent_avg = statistics.mean(numbers[-len(numbers)//4:])
                historical_avg = statistics.mean(numbers[:len(numbers)//2])
                momentum = "accelerating" if recent_avg > historical_avg else "decelerating"
                result["momentum"] = {
                    "status": momentum,
                    "recent_average": round(recent_avg, 4),
                    "historical_average": round(historical_avg, 4)
                }
            
            # Volatility
            if len(changes) > 1:
                result["volatility"] = {
                    "std_of_changes": round(statistics.stdev(changes), 4),
                    "max_increase": round(max(changes), 4),
                    "max_decrease": round(min(changes), 4)
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Trend analysis failed: {str(e)}"}
    
    def _comparison_analysis(self, data: List) -> Dict[str, Any]:
        """Compare multiple data series or groups"""
        try:
            if isinstance(data, dict):
                # Compare named groups
                results = {}
                for name, values in data.items():
                    numbers = self._extract_numbers(values)
                    if numbers:
                        results[name] = {
                            "mean": round(statistics.mean(numbers), 4),
                            "median": statistics.median(numbers),
                            "count": len(numbers),
                            "sum": sum(numbers)
                        }
                
                if len(results) >= 2:
                    means = {k: v["mean"] for k, v in results.items()}
                    highest = max(means, key=lambda k: means[k])
                    lowest = min(means, key=lambda k: means[k])
                    
                    return {
                        "comparison": {
                            "groups": results,
                            "highest_mean": {"group": highest, "value": means[highest]},
                            "lowest_mean": {"group": lowest, "value": means[lowest]},
                            "difference": round(means[highest] - means[lowest], 4)
                        }
                    }
                
                return {"comparison": {"groups": results}}
                
            else:
                # Treat as single series, compare quarters
                numbers = self._extract_numbers(data)
                if len(numbers) < 4:
                    return {"error": "Need at least 4 data points for comparison"}
                
                quarter_size = len(numbers) // 4
                quarters = {
                    "Q1": numbers[:quarter_size],
                    "Q2": numbers[quarter_size:2*quarter_size],
                    "Q3": numbers[2*quarter_size:3*quarter_size],
                    "Q4": numbers[3*quarter_size:]
                }
                
                comparison = {}
                for q, vals in quarters.items():
                    if vals:
                        comparison[q] = {
                            "mean": round(statistics.mean(vals), 4),
                            "sum": sum(vals)
                        }
                
                return {"comparison": {"quarters": comparison}}
                
        except Exception as e:
            return {"error": f"Comparison analysis failed: {str(e)}"}
    
    def _summary_analysis(self, data: List) -> Dict[str, Any]:
        """Generate a comprehensive summary of the data"""
        try:
            numbers = self._extract_numbers(data)
            
            result = {
                "summary": {
                    "total_items": len(data),
                    "data_type": self._detect_data_type(data)
                }
            }
            
            if numbers:
                result["summary"]["numeric_values"] = len(numbers)
                result["summary"]["total"] = sum(numbers)
                result["summary"]["average"] = round(statistics.mean(numbers), 4)
                result["summary"]["range"] = {
                    "min": min(numbers),
                    "max": max(numbers)
                }
                
                # Quick health check
                if len(numbers) > 1:
                    std = statistics.stdev(numbers)
                    mean = statistics.mean(numbers)
                    cv = (std / mean * 100) if mean != 0 else 0
                    
                    result["data_quality"] = {
                        "coefficient_of_variation": round(cv, 2),
                        "variability": "high" if cv > 50 else "moderate" if cv > 20 else "low"
                    }
            
            return result
            
        except Exception as e:
            return {"error": f"Summary analysis failed: {str(e)}"}
    
    def _extract_numbers(self, data: Any) -> List[float]:
        """Extract numerical values from various data formats"""
        numbers = []
        
        if isinstance(data, (int, float)):
            return [float(data)]
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (int, float)):
                    numbers.append(float(item))
                elif isinstance(item, dict):
                    # Extract numeric values from dict
                    for v in item.values():
                        if isinstance(v, (int, float)):
                            numbers.append(float(v))
                elif isinstance(item, str):
                    try:
                        numbers.append(float(item))
                    except ValueError:
                        pass
        
        return numbers
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data"""
        if not data:
            return "empty"
        
        if isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                return "numeric_array"
            if all(isinstance(x, str) for x in data):
                return "string_array"
            if all(isinstance(x, dict) for x in data):
                return "object_array"
            return "mixed_array"
        
        return type(data).__name__
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        # Based on statistics
        if "statistics" in insights:
            stats = insights["statistics"]
            if "std_dev" in stats and stats.get("mean", 0) != 0:
                cv = stats["std_dev"] / stats["mean"] * 100
                if cv > 50:
                    recommendations.append("High variability detected. Consider investigating causes of variation.")
        
        # Based on outliers
        if "outliers" in insights:
            outlier_count = insights["outliers"].get("count", 0)
            if outlier_count > 0:
                recommendations.append(f"Found {outlier_count} outliers. Review these data points for errors or special cases.")
        
        # Based on trend
        if "trend" in insights:
            trend = insights["trend"]
            if trend.get("direction") == "decreasing":
                recommendations.append("Downward trend detected. Investigate potential causes and consider corrective actions.")
            if trend.get("percent_change", 0) > 50:
                recommendations.append("Significant change detected (>50%). This may require immediate attention.")
        
        # Based on momentum
        if "momentum" in insights:
            if insights["momentum"].get("status") == "accelerating":
                recommendations.append("Momentum is accelerating. Consider capitalizing on this trend.")
            elif insights["momentum"].get("status") == "decelerating":
                recommendations.append("Momentum is slowing. Plan for potential stabilization or reversal.")
        
        if not recommendations:
            recommendations.append("Data appears stable. Continue monitoring for changes.")
        
        return recommendations

    async def _arun(self, input_str: str) -> str:
        """Async implementation"""
        return self._run(input_str)
