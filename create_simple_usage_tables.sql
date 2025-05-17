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

-- Add comment to tables for documentation
COMMENT ON TABLE public.organization_usage IS 'Aggregated usage tracking for organizations (daily, weekly, monthly, etc.)';
COMMENT ON TABLE public.usage_detail IS 'Detailed usage tracking for audit and analysis purposes'; 