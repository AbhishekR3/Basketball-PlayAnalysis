--/*
CREATE OR REPLACE PROCEDURE closest_basketball(top_n INTEGER DEFAULT 3)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$

DECLARE
    frame_count INTEGER;
    
BEGIN
    -- Create a table to store the results
    DROP TABLE IF EXISTS public.closest_players;
    CREATE TABLE public.closest_players AS
    WITH 

    -- Get the position of the basketball for each frame
    basketball_position AS (
        SELECT frame, point_geom
        FROM tracking_data
        WHERE is_basketball = TRUE
    ),

    -- Calculate distances between basketball and players
    player_distances AS (
        SELECT 
            b.frame,
            p.id AS player_id,
            ST_Distance(p.point_geom, b.point_geom) AS distance,
            ROW_NUMBER() OVER (PARTITION BY b.frame ORDER BY ST_Distance(p.point_geom, b.point_geom)) AS rank
        FROM 
            basketball_position b
            JOIN tracking_data p ON b.frame = p.frame
        WHERE 
            p.is_basketball = FALSE
    )

    -- Select top N closest players for each frame
    SELECT 
        frame,
        player_id,
        ROUND(distance::numeric, 8) AS distance,
        rank
    FROM 
        player_distances
    WHERE 
        rank <= top_n
    ORDER BY 
        frame, rank;

    -- Get the count of distinct frames
    SELECT COUNT(DISTINCT frame) INTO frame_count FROM public.closest_players;

    -- Print the number of records processed
    RAISE NOTICE 'Processed % frames, stored results in public.closest_players table', frame_count;
END;
$$;
--*/

call closest_basketball(3)