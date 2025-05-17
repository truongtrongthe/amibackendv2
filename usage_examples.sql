-- Example data insertion for testing
-- Replace these UUIDs with your actual organization IDs
INSERT INTO public.organization_usage (org_id, type, count, date)
VALUES 
    ('11111111-1111-1111-1111-111111111111', 'message', 50, CURRENT_DATE),
    ('11111111-1111-1111-1111-111111111111', 'reasoning', 30, CURRENT_DATE),
    ('11111111-1111-1111-1111-111111111111', 'message', 45, CURRENT_DATE - INTERVAL '1 day'),
    ('11111111-1111-1111-1111-111111111111', 'reasoning', 25, CURRENT_DATE - INTERVAL '1 day'),
    ('22222222-2222-2222-2222-222222222222', 'message', 120, CURRENT_DATE),
    ('22222222-2222-2222-2222-222222222222', 'reasoning', 75, CURRENT_DATE);

-- Example detailed usage records
INSERT INTO public.usage_detail (org_id, type, count, timestamp)
VALUES 
    ('11111111-1111-1111-1111-111111111111', 'message', 1, NOW()),
    ('11111111-1111-1111-1111-111111111111', 'message', 5, NOW() - INTERVAL '30 minutes'),
    ('11111111-1111-1111-1111-111111111111', 'reasoning', 2, NOW() - INTERVAL '1 hour'),
    ('22222222-2222-2222-2222-222222222222', 'message', 3, NOW()),
    ('22222222-2222-2222-2222-222222222222', 'reasoning', 1, NOW() - INTERVAL '15 minutes');

-- Example: Use the increment_organization_usage function
SELECT increment_organization_usage(
    '11111111-1111-1111-1111-111111111111',
    'message',
    10
);

-- Example: Get usage summary for the last 7 days
SELECT * FROM get_organization_usage_summary(
    '11111111-1111-1111-1111-111111111111',
    CURRENT_DATE - INTERVAL '7 days',
    CURRENT_DATE
);

-- Example queries for usage analysis

-- 1. Daily usage by type for a specific organization
SELECT 
    date,
    type,
    count
FROM 
    public.organization_usage
WHERE 
    org_id = '11111111-1111-1111-1111-111111111111'
    AND date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY 
    date DESC, type;

-- 2. Weekly usage aggregation
SELECT 
    date_trunc('week', date) as week,
    type,
    SUM(count) as total
FROM 
    public.organization_usage
WHERE 
    org_id = '11111111-1111-1111-1111-111111111111'
    AND date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY 
    week, type
ORDER BY 
    week DESC, type;

-- 3. Monthly usage comparison
SELECT 
    date_trunc('month', date) as month,
    type,
    SUM(count) as total
FROM 
    public.organization_usage
WHERE 
    org_id = '11111111-1111-1111-1111-111111111111'
    AND date >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY 
    month, type
ORDER BY 
    month DESC, type;

-- 4. Usage comparison across organizations
SELECT 
    org_id,
    type,
    SUM(count) as total_count,
    AVG(count) as avg_daily_count,
    MAX(count) as max_daily_count
FROM 
    public.organization_usage
WHERE 
    date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY 
    org_id, type
ORDER BY 
    total_count DESC;

-- 5. Hourly usage patterns from detailed logs
SELECT 
    date_trunc('hour', timestamp) as hour,
    type,
    SUM(count) as total
FROM 
    public.usage_detail
WHERE 
    org_id = '11111111-1111-1111-1111-111111111111'
    AND timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY 
    hour, type
ORDER BY 
    hour DESC, type;

-- 6. Message to reasoning ratio
WITH usage_totals AS (
    SELECT
        org_id,
        type,
        SUM(count) as total
    FROM
        public.organization_usage
    WHERE
        date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY
        org_id, type
)
SELECT
    org_id,
    MAX(CASE WHEN type = 'message' THEN total ELSE 0 END) as message_count,
    MAX(CASE WHEN type = 'reasoning' THEN total ELSE 0 END) as reasoning_count,
    ROUND(
        MAX(CASE WHEN type = 'message' THEN total ELSE 0 END)::numeric / 
        NULLIF(MAX(CASE WHEN type = 'reasoning' THEN total ELSE 0 END), 0)::numeric,
        2
    ) as message_reasoning_ratio
FROM
    usage_totals
GROUP BY
    org_id
ORDER BY
    message_count DESC; 