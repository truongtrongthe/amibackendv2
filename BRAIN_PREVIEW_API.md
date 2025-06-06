# Brain Preview API Documentation

## Endpoint Overview

The `/brain-preview` endpoint provides comprehensive analysis of the AI knowledge base, including statistics, insights, and AI-generated summaries of all stored knowledge.

## 📋 Endpoint Details

- **URL**: `/brain-preview`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Response-Type**: `application/json`

## 🔧 Request Schema

### Request Body

```typescript
interface BrainPreviewRequest {
  namespace?: string;           // Default: "conversation"
  max_vectors?: number;         // Default: 1000, Max recommended: 10000
  summary_type?: "comprehensive" | "topics_only" | "categories_only"; // Default: "comprehensive"
  include_vectors?: boolean;    // Default: false (WARNING: Large responses)
  max_content_length?: number;  // Default: 50000 (for LLM analysis)
}
```

### Example Requests

#### Basic Request (Recommended)
```json
{
  "namespace": "conversation",
  "max_vectors": 1000,
  "summary_type": "comprehensive",
  "include_vectors": false
}
```

#### Minimal Request (All defaults)
```json
{}
```

#### Advanced Request with Vector Details
```json
{
  "namespace": "conversation", 
  "max_vectors": 500,
  "summary_type": "topics_only",
  "include_vectors": true,
  "max_content_length": 30000
}
```

## 📤 Response Schema

### Success Response (200)

```typescript
interface BrainPreviewResponse {
  success: true;
  request_id: string;
  namespace: string;
  generated_at: string;           // ISO timestamp
  processing_time_seconds: number;
  
  vector_analysis: {
    total_ai_synthesis_vectors: number;
    vectors_analyzed: number;
    statistics: {
      categories: {
        unique_count: number;
        top_10: Record<string, number>;
        total_category_instances: number;
      };
      topics: {
        unique_count: number;
        top_10: Record<string, number>;
      };
      users: {
        unique_count: number;
        top_10: Record<string, number>;
      };
      threads: {
        unique_count: number;
        total_threads_with_ai_synthesis: number;
      };
      content: {
        average_length: number;
        min_length: number;
        max_length: number;
      };
      confidence: {
        average: number;
        min: number;
        max: number;
      };
      dates?: {
        earliest: string;
        latest: string;
        span_days: number;
      };
    };
    query_params: {
      batch_size: number;
      max_vectors: number | null;
      actual_fetched: number;
    };
  };
  
  knowledge_summary?: {
    summary_type: string;
    analysis: {
      success: boolean;
      analysis: string;              // LLM-generated comprehensive analysis
      content_length_analyzed: number;
      generated_at: string;
    };
    content_organization: {
      categories_count: number;
      topics_count: number;
      content_sections: number;
    };
    processing_info: {
      max_content_length: number;
      actual_content_length: number;
      truncated: boolean;
    };
  };
  
  insights: {
    knowledge_health: {
      total_vectors: number;
      average_content_length: number;
      average_confidence: number;
      category_diversity: number;
      topic_diversity: number;
    };
    recommendations: string[];
  };
  
  vectors?: {                      // Only if include_vectors: true
    count: number;
    total_available: number;
    vectors: Array<{
      id: string;
      title: string;
      created_at: string;
      confidence: number;
      categories: string[];
      topic: string;
      content_preview: string;     // Truncated to 300 chars
    }>;
  };
}
```

### Error Response (500)

```typescript
interface BrainPreviewErrorResponse {
  detail: string;
}
```

## 💻 Frontend Implementation Examples

### React Hook Example

```tsx
import { useState } from 'react';

interface BrainPreviewRequest {
  namespace?: string;
  max_vectors?: number;
  summary_type?: 'comprehensive' | 'topics_only' | 'categories_only';
  include_vectors?: boolean;
  max_content_length?: number;
}

export const useBrainPreview = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchBrainPreview = async (params: BrainPreviewRequest = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/brain-preview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          namespace: 'conversation',
          max_vectors: 1000,
          summary_type: 'comprehensive',
          include_vectors: false,
          ...params
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, fetchBrainPreview };
};
```

### Vanilla JavaScript Example

```javascript
async function getBrainPreview(options = {}) {
  const defaultOptions = {
    namespace: 'conversation',
    max_vectors: 1000,
    summary_type: 'comprehensive',
    include_vectors: false
  };

  try {
    const response = await fetch('/brain-preview', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...defaultOptions, ...options })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Brain preview fetch failed:', error);
    throw error;
  }
}

// Usage examples
getBrainPreview()
  .then(data => {
    console.log('Knowledge stats:', data.vector_analysis.statistics);
    console.log('AI insights:', data.knowledge_summary?.analysis);
    console.log('Recommendations:', data.insights.recommendations);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });
```

### TypeScript Fetch Function

```typescript
type BrainPreviewOptions = {
  namespace?: string;
  maxVectors?: number;
  summaryType?: 'comprehensive' | 'topics_only' | 'categories_only';
  includeVectors?: boolean;
  maxContentLength?: number;
};

export async function fetchBrainPreview(
  options: BrainPreviewOptions = {}
): Promise<BrainPreviewResponse> {
  const requestBody = {
    namespace: options.namespace ?? 'conversation',
    max_vectors: options.maxVectors ?? 1000,
    summary_type: options.summaryType ?? 'comprehensive',
    include_vectors: options.includeVectors ?? false,
    max_content_length: options.maxContentLength ?? 50000
  };

  const response = await fetch('/brain-preview', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}
```

## 🎯 Use Cases & Examples

### Dashboard Overview
```javascript
// Get basic knowledge stats for dashboard
const dashboardData = await getBrainPreview({
  max_vectors: 500,
  summary_type: 'comprehensive',
  include_vectors: false
});

// Display metrics
const stats = dashboardData.vector_analysis.statistics;
console.log(`Total Knowledge Entries: ${stats.total_vectors}`);
console.log(`Categories: ${stats.categories.unique_count}`);
console.log(`Average Confidence: ${(stats.confidence.average * 100).toFixed(1)}%`);
```

### Knowledge Explorer
```javascript
// Get detailed vectors for exploration
const explorerData = await getBrainPreview({
  max_vectors: 100,
  summary_type: 'topics_only',
  include_vectors: true
});

// Process vectors for display
explorerData.vectors?.vectors.forEach(vector => {
  console.log(`${vector.title}: ${vector.content_preview}`);
});
```

### Knowledge Health Check
```javascript
// Get recommendations for knowledge base improvement
const healthCheck = await getBrainPreview({
  max_vectors: 2000,
  summary_type: 'comprehensive'
});

// Display recommendations
healthCheck.insights.recommendations.forEach(rec => {
  console.log(`💡 ${rec}`);
});
```

## ⚠️ Important Considerations

### Performance
- **Response Time**: Typically 2-10 seconds depending on knowledge base size
- **Large Responses**: Setting `include_vectors: true` can result in large responses
- **Rate Limiting**: Recommended max 1 request per 30 seconds for comprehensive analysis

### Memory Usage
- Large knowledge bases (>5000 vectors) may take longer to process
- Consider using smaller `max_vectors` values for faster responses
- The `max_content_length` parameter controls LLM analysis scope

### Error Handling
```javascript
try {
  const data = await getBrainPreview();
  // Handle success
} catch (error) {
  if (error.message.includes('500')) {
    // Server error - possibly knowledge base issue
    console.error('Knowledge base analysis failed');
  } else if (error.message.includes('timeout')) {
    // Network timeout
    console.error('Request timed out - try with smaller max_vectors');
  } else {
    // Other errors
    console.error('Unexpected error:', error.message);
  }
}
```

## 🔄 Response Data Usage Examples

### Extract Key Metrics
```javascript
function extractKeyMetrics(brainData) {
  const { vector_analysis, insights } = brainData;
  
  return {
    totalKnowledge: vector_analysis.total_ai_synthesis_vectors,
    processingTime: brainData.processing_time_seconds,
    healthScore: insights.knowledge_health.average_confidence,
    categoryDiversity: insights.knowledge_health.category_diversity,
    topCategories: Object.keys(vector_analysis.statistics.categories.top_10),
    recommendations: insights.recommendations
  };
}
```

### Format for UI Display
```javascript
function formatBrainDataForUI(brainData) {
  const stats = brainData.vector_analysis.statistics;
  
  return {
    overview: {
      totalEntries: stats.total_vectors,
      avgLength: Math.round(stats.content.average_length),
      confidenceScore: `${(stats.confidence.average * 100).toFixed(1)}%`,
      categories: stats.categories.unique_count,
      timeSpan: stats.dates ? `${stats.dates.span_days} days` : 'N/A'
    },
    insights: brainData.knowledge_summary?.analysis?.analysis || 'No analysis available',
    recommendations: brainData.insights.recommendations
  };
}
```

## 🚀 Quick Start

1. **Basic Implementation**:
```javascript
fetch('/brain-preview', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({})
}).then(r => r.json()).then(console.log);
```

2. **Add Error Handling**:
```javascript
try {
  const response = await fetch('/brain-preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ max_vectors: 500 })
  });
  const data = await response.json();
  // Use data...
} catch (error) {
  console.error('Failed to fetch brain preview:', error);
}
```

3. **Extract What You Need**:
```javascript
const { vector_analysis, insights, knowledge_summary } = data;
// Display statistics, recommendations, and AI analysis
``` 