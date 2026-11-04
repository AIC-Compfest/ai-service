-- Create ai_processing_jobs table for background job queue
CREATE TABLE IF NOT EXISTS ai_processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'QUEUED',
    file_storage_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    result_data JSONB,
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED'))
);

-- Create index for efficient job polling
CREATE INDEX idx_jobs_status_created ON ai_processing_jobs(status, created_at);

-- Create index for document lookup
CREATE INDEX idx_jobs_document_id ON ai_processing_jobs(document_id);

-- Enable Row Level Security (optional, but recommended)
ALTER TABLE ai_processing_jobs ENABLE ROW LEVEL SECURITY;

-- Create a policy for service role access (full access)
CREATE POLICY "Service role can do anything" ON ai_processing_jobs
    FOR ALL
    USING (auth.role() = 'service_role');
