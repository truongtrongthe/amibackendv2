-- Create organization_usage table for aggregated usage tracking
CREATE TABLE IF NOT EXISTS public.organization_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Enforce uniqueness of (org_id, type, date) to prevent duplicates
    CONSTRAINT unique_org_type_date UNIQUE (org_id, type, date)
);

-- Create usage_detail table for granular usage tracking
CREATE TABLE IF NOT EXISTS public.usage_detail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add indexes for fast querying
CREATE INDEX IF NOT EXISTS organization_usage_org_id_idx ON public.organization_usage (org_id);
CREATE INDEX IF NOT EXISTS organization_usage_date_idx ON public.organization_usage (date);
CREATE INDEX IF NOT EXISTS organization_usage_type_idx ON public.organization_usage (type);
CREATE INDEX IF NOT EXISTS organization_usage_org_date_idx ON public.organization_usage (org_id, date);
CREATE INDEX IF NOT EXISTS organization_usage_org_type_idx ON public.organization_usage (org_id, type);

CREATE INDEX IF NOT EXISTS usage_detail_org_id_idx ON public.usage_detail (org_id);
CREATE INDEX IF NOT EXISTS usage_detail_timestamp_idx ON public.usage_detail (timestamp);
CREATE INDEX IF NOT EXISTS usage_detail_type_idx ON public.usage_detail (type);
CREATE INDEX IF NOT EXISTS usage_detail_org_timestamp_idx ON public.usage_detail (org_id, timestamp);

-- Add RLS (Row Level Security) policies
ALTER TABLE public.organization_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.usage_detail ENABLE ROW LEVEL SECURITY;

-- Create policy to allow organizations to access only their own usage data
CREATE POLICY organization_usage_policy ON public.organization_usage 
    FOR ALL
    USING (auth.uid() = org_id);

CREATE POLICY usage_detail_policy ON public.usage_detail
    FOR ALL
    USING (auth.uid() = org_id);

-- Create trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_organization_usage_updated_at
BEFORE UPDATE ON public.organization_usage
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Create or replace function to increment usage count
CREATE OR REPLACE FUNCTION increment_organization_usage(
    p_org_id UUID,
    p_type VARCHAR,
    p_count INTEGER,
    p_date DATE DEFAULT CURRENT_DATE
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO public.organization_usage (org_id, type, count, date)
    VALUES (p_org_id, p_type, p_count, p_date)
    ON CONFLICT (org_id, type, date)
    DO UPDATE SET
        count = public.organization_usage.count + p_count,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Add comment to tables for documentation
COMMENT ON TABLE public.organization_usage IS 'Aggregated usage tracking for organizations (daily, weekly, monthly, etc.)';
COMMENT ON TABLE public.usage_detail IS 'Detailed usage tracking for audit and analysis purposes';

-- Example stored procedure for common usage reports
CREATE OR REPLACE FUNCTION get_organization_usage_summary(
    p_org_id UUID,
    p_start_date DATE DEFAULT (CURRENT_DATE - INTERVAL '30 days'),
    p_end_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    usage_type VARCHAR,
    total_count BIGINT,
    daily_average NUMERIC(10,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        type as usage_type,
        SUM(count) as total_count,
        ROUND(AVG(count)::numeric, 2) as daily_average
    FROM
        public.organization_usage
    WHERE
        org_id = p_org_id AND
        date BETWEEN p_start_date AND p_end_date
    GROUP BY
        type
    ORDER BY
        total_count DESC;
END;
$$ LANGUAGE plpgsql; 