ALTER TABLE conversations ADD COLUMN initiator TEXT;

-- Backfill from existing context JSON data
UPDATE conversations SET initiator = 'agent'
WHERE context LIKE '%"initiator":"agent"%';

-- Also mark conversations with parent_id as agent-initiated
UPDATE conversations SET initiator = 'agent'
WHERE parent_id IS NOT NULL AND initiator IS NULL;
