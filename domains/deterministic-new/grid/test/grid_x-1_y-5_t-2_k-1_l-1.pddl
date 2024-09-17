(define (problem grid-x1-y5-t2-k10-l10-p100)
(:domain grid)
(:objects 
        f0-0f 
        f0-1f 
        f0-2f 
        f0-3f 
        f0-4f 
        shape0 shape1 
        key0-0 
)
(:init
(arm-empty)
(place f0-0f)
(place f0-1f)
(place f0-2f)
(place f0-3f)
(place f0-4f)
(shape shape0)
(shape shape1)
(key key0-0)
(key-shape key0-0 shape0)
(conn f0-0f f0-1f)
(conn f0-1f f0-2f)
(conn f0-2f f0-3f)
(conn f0-3f f0-4f)
(conn f0-1f f0-0f)
(conn f0-2f f0-1f)
(conn f0-3f f0-2f)
(conn f0-4f f0-3f)
(open f0-0f)
(open f0-2f)
(open f0-3f)
(open f0-4f)
(locked f0-1f)
(lock-shape f0-1f shape0)
(at key0-0 f0-0f)
(at-robot f0-4f)
)
(:goal
(and
(at key0-0 f0-1f)
)
)
)
