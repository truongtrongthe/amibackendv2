-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Brain Graph table
CREATE TABLE brain_graph (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    graph_id BIGSERIAL,  -- Auto-incrementing integer ID
    org_id UUID NOT NULL UNIQUE,  -- Each org has exactly one brain graph
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (org_id) REFERENCES organization(id) ON DELETE CASCADE
);

-- Brain Graph Version table
CREATE TABLE brain_graph_version (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    version_id BIGSERIAL,  -- Auto-incrementing integer ID
    graph_id UUID NOT NULL,
    version_number INTEGER NOT NULL,
    brain_ids INTEGER[] NOT NULL DEFAULT '{}',  -- Array of brain IDs
    released_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL CHECK (status IN ('training', 'published')),
    FOREIGN KEY (graph_id) REFERENCES brain_graph(id) ON DELETE CASCADE,
    UNIQUE (graph_id, version_number)  -- Ensure version numbers are unique per graph
);

-- Create indexes for better query performance
CREATE INDEX idx_brain_graph_org_id ON brain_graph(org_id);
CREATE INDEX idx_brain_graph_version_graph_id ON brain_graph_version(graph_id);
CREATE INDEX idx_brain_graph_version_status ON brain_graph_version(status);

-- Create a function to auto-increment version number per graph
CREATE OR REPLACE FUNCTION next_version_number(graph_uuid UUID)
RETURNS INTEGER AS $$
DECLARE
    next_version INTEGER;
BEGIN
    SELECT COALESCE(MAX(version_number) + 1, 1)
    INTO next_version
    FROM brain_graph_version
    WHERE graph_id = graph_uuid;
    RETURN next_version;
END;
$$ LANGUAGE plpgsql; 