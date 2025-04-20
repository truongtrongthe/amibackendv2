-- Create Contact Analysis Table: Store sales signal analysis for contacts

CREATE TABLE IF NOT EXISTS contact_analysis (
    id SERIAL PRIMARY KEY,
    contact_id INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    sales_signals JSONB NOT NULL,
    score_breakdown JSONB NOT NULL,
    sales_readiness_score INTEGER NOT NULL,
    priority TEXT NOT NULL,
    analyzed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_contact_analysis_contact_id ON contact_analysis(contact_id);
CREATE INDEX IF NOT EXISTS idx_contact_analysis_organization_id ON contact_analysis(organization_id);
CREATE INDEX IF NOT EXISTS idx_contact_analysis_priority ON contact_analysis(priority);
CREATE INDEX IF NOT EXISTS idx_contact_analysis_sales_readiness_score ON contact_analysis(sales_readiness_score);

COMMENT ON TABLE contact_analysis IS 'Stores sales signal analysis for contacts';
COMMENT ON COLUMN contact_analysis.contact_id IS 'The contact this analysis belongs to';
COMMENT ON COLUMN contact_analysis.organization_id IS 'The organization this contact belongs to';
COMMENT ON COLUMN contact_analysis.sales_signals IS 'JSON array of detected sales signals';
COMMENT ON COLUMN contact_analysis.score_breakdown IS 'JSON object with scores for different categories';
COMMENT ON COLUMN contact_analysis.sales_readiness_score IS 'Overall numerical sales readiness score (0-100)';
COMMENT ON COLUMN contact_analysis.priority IS 'Priority level (High, Medium, Low, Very Low)';
COMMENT ON COLUMN contact_analysis.analyzed_at IS 'When the analysis was performed';
COMMENT ON COLUMN contact_analysis.created_at IS 'When this record was created';

-- Create a function to store analysis history
CREATE OR REPLACE FUNCTION store_contact_analysis(
    p_contact_id INTEGER,
    p_organization_id UUID,
    p_sales_signals JSONB,
    p_score_breakdown JSONB,
    p_sales_readiness_score INTEGER,
    p_priority TEXT,
    p_analyzed_at TIMESTAMP WITH TIME ZONE
) RETURNS INTEGER AS $$
DECLARE
    inserted_id INTEGER;
BEGIN
    INSERT INTO contact_analysis (
        contact_id,
        organization_id,
        sales_signals,
        score_breakdown,
        sales_readiness_score,
        priority,
        analyzed_at
    ) VALUES (
        p_contact_id,
        p_organization_id,
        p_sales_signals,
        p_score_breakdown,
        p_sales_readiness_score,
        p_priority,
        p_analyzed_at
    )
    RETURNING id INTO inserted_id;
    
    RETURN inserted_id;
END;
$$ LANGUAGE plpgsql; 