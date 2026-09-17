(define (problem grid-x1-y7-t2-k40-l20-p100)
(:domain grid)
(:objects 
        f0-0f 
        f0-1f 
        f0-2f 
        f0-3f 
        f0-4f 
        f0-5f 
        f0-6f 
        shape0 shape1 
        key0-0 key0-1 key0-2 key0-3 
)
(:init
(arm-empty)
(place f0-0f)
(place f0-1f)
(place f0-2f)
(place f0-3f)
(place f0-4f)
(place f0-5f)
(place f0-6f)
(shape shape0)
(shape shape1)
(key key0-0)
(key-shape key0-0 shape0)
(key key0-1)
(key-shape key0-1 shape0)
(key key0-2)
(key-shape key0-2 shape0)
(key key0-3)
(key-shape key0-3 shape0)
(conn f0-0f f0-1f)
(conn f0-1f f0-2f)
(conn f0-2f f0-3f)
(conn f0-3f f0-4f)
(conn f0-4f f0-5f)
(conn f0-5f f0-6f)
(conn f0-1f f0-0f)
(conn f0-2f f0-1f)
(conn f0-3f f0-2f)
(conn f0-4f f0-3f)
(conn f0-5f f0-4f)
(conn f0-6f f0-5f)
(open f0-0f)
(open f0-1f)
(open f0-2f)
(open f0-5f)
(open f0-6f)
(locked f0-3f)
(lock-shape f0-3f shape0)
(locked f0-4f)
(lock-shape f0-4f shape0)
(at key0-0 f0-6f)
(at key0-1 f0-2f)
(at key0-2 f0-4f)
(at key0-3 f0-0f)
(at-robot f0-1f)
)
(:goal
(and
(at key0-0 f0-3f)
(at key0-1 f0-0f)
(at key0-2 f0-6f)
(at key0-3 f0-4f)
)
)
)
