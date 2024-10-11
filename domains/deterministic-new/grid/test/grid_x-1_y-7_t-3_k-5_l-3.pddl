(define (problem grid-x1-y7-t3-k50-l30-p100)
(:domain grid)
(:objects 
        f0-0f 
        f0-1f 
        f0-2f 
        f0-3f 
        f0-4f 
        f0-5f 
        f0-6f 
        shape0 shape1 shape2 
        key1-0 key1-1 key1-2 key1-3 key1-4 
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
(shape shape2)
(key key1-0)
(key-shape key1-0 shape1)
(key key1-1)
(key-shape key1-1 shape1)
(key key1-2)
(key-shape key1-2 shape1)
(key key1-3)
(key-shape key1-3 shape1)
(key key1-4)
(key-shape key1-4 shape1)
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
(open f0-1f)
(open f0-2f)
(open f0-3f)
(open f0-6f)
(locked f0-5f)
(lock-shape f0-5f shape1)
(locked f0-4f)
(lock-shape f0-4f shape1)
(locked f0-0f)
(lock-shape f0-0f shape1)
(at key1-0 f0-2f)
(at key1-1 f0-2f)
(at key1-2 f0-4f)
(at key1-3 f0-3f)
(at key1-4 f0-1f)
(at-robot f0-6f)
)
(:goal
(and
(at key1-0 f0-1f)
(at key1-1 f0-0f)
(at key1-2 f0-2f)
(at key1-3 f0-4f)
(at key1-4 f0-4f)
)
)
)