(define (problem grid-x1-y3-t2-k20-l20-p100)
(:domain grid)
(:objects 
        f0-0f 
        f0-1f 
        f0-2f 
        shape0 shape1 
        key0-0 key0-1 
)
(:init
(arm-empty)
(place f0-0f)
(place f0-1f)
(place f0-2f)
(shape shape0)
(shape shape1)
(key key0-0)
(key-shape key0-0 shape0)
(key key0-1)
(key-shape key0-1 shape0)
(conn f0-0f f0-1f)
(conn f0-1f f0-2f)
(conn f0-1f f0-0f)
(conn f0-2f f0-1f)
(open f0-0f)
(locked f0-2f)
(lock-shape f0-2f shape0)
(locked f0-1f)
(lock-shape f0-1f shape0)
(at key0-0 f0-1f)
(at key0-1 f0-2f)
(at-robot f0-0f)
)
(:goal
(and
(at key0-0 f0-1f)
(at key0-1 f0-0f)
)
)
)
