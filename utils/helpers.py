"""
Helper utility functions for the research Q/A bot
"""

import re
import string
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from config import settings


def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user query
    
    Args:
        query: User input query
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    query = query.strip()
    
    if len(query) < settings.MIN_QUERY_LENGTH:
        return False, f"Query must be at least {settings.MIN_QUERY_LENGTH} characters long"
    
    if len(query) > settings.MAX_QUERY_LENGTH:
        return False, f"Query must not exceed {settings.MAX_QUERY_LENGTH} characters"
    
    # Check for only special characters
    if not re.search(r'[a-zA-Z0-9]', query):
        return False, "Query must contain at least some alphanumeric characters"
    
    return True, None


def format_timestamp(dt: datetime = None, format_type: str = "full") -> str:
    """
    Format timestamp for display
    
    Args:
        dt: Datetime object (defaults to now)
        format_type: Type of format ("full", "date", "time")
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    
    if format_type == "full":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "date":
        return dt.strftime("%Y-%m-%d")
    elif format_type == "time":
        return dt.strftime("%H:%M:%S")
    elif format_type == "short":
        return dt.strftime("%m/%d %H:%M")
    else:
        return dt.isoformat()


def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Whether to add "..." at the end
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length].rstrip()
    
    if add_ellipsis:
        truncated += "..."
    
    return truncated


def calculate_confidence(scores: List[float], weights: List[float] = None) -> float:
    """
    Calculate overall confidence from multiple scores
    
    Args:
        scores: List of confidence scores (0.0 to 1.0)
        weights: Optional weights for each score
        
    Returns:
        Weighted average confidence score
    """
    if not scores:
        return 0.0
    
    # Filter out invalid scores
    valid_scores = [s for s in scores if 0.0 <= s <= 1.0]
    
    if not valid_scores:
        return 0.0
    
    if weights is None:
        # Equal weights
        return sum(valid_scores) / len(valid_scores)
    
    if len(weights) != len(valid_scores):
        # Fall back to equal weights if mismatch
        return sum(valid_scores) / len(valid_scores)
    
    # Weighted average
    weighted_sum = sum(score * weight for score, weight in zip(valid_scores, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove invalid characters
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)
    
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # Ensure it doesn't start/end with special chars
    filename = filename.strip('_.-')
    
    return filename if filename else "untitled"


def format_processing_time(seconds: float) -> str:
    """
    Format processing time for display
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple implementation)
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Simple keyword extraction
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words
    words = text.split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those'
    }
    
    # Filter words
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # Count frequency
    word_count = {}
    for word in keywords:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, count in sorted_words[:max_keywords]]


def format_authors_list(authors: List[str], max_display: int = 3) -> str:
    """
    Format authors list for display
    
    Args:
        authors: List of author names
        max_display: Maximum number of authors to display
        
    Returns:
        Formatted authors string
    """
    if not authors:
        return "Unknown Authors"
    
    if len(authors) <= max_display:
        return ", ".join(authors)
    
    displayed = authors[:max_display]
    remaining = len(authors) - max_display
    
    return f"{', '.join(displayed)} et al. (+{remaining} more)"


def estimate_reading_time(text: str, words_per_minute: int = 200) -> str:
    """
    Estimate reading time for text
    
    Args:
        text: Text content
        words_per_minute: Average reading speed
        
    Returns:
        Formatted reading time estimate
    """
    if not text:
        return "0 min"
    
    word_count = len(text.split())
    minutes = word_count / words_per_minute
    
    if minutes < 1:
        return "< 1 min"
    elif minutes < 60:
        return f"{minutes:.0f} min"
    else:
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        return f"{hours}h {remaining_minutes}m"


def create_query_suggestions(original_query: str, context: List[str] = None) -> List[str]:
    """
    Create query suggestions based on original query
    
    Args:
        original_query: Original user query
        context: Additional context for suggestions
        
    Returns:
        List of suggested queries
    """
    suggestions = []
    
    # Extract keywords from original query
    keywords = extract_keywords(original_query, max_keywords=3)
    
    if keywords:
        # Create variations
        suggestions.extend([
            f"{' '.join(keywords)} methodology",
            f"{' '.join(keywords)} applications",
            f"{' '.join(keywords)} limitations",
            f"recent advances in {' '.join(keywords)}",
            f"comparison of {' '.join(keywords)} approaches"
        ])
    
    # Add context-based suggestions if available
    if context:
        context_keywords = []
        for item in context[:2]:  # Use first 2 context items
            context_keywords.extend(extract_keywords(item, max_keywords=2))
        
        if context_keywords:
            suggestions.append(f"{' '.join(context_keywords[:3])} review")
    
    return suggestions[:5]  # Return top 5 suggestions
