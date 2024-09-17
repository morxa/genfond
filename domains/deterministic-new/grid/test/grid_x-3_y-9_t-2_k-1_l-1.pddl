(define (problem grid-x3-y9-t2-k10-l10-p100)
(:domain grid)
(:objects 
        f0-0f f1-0f f2-0f 
        f0-1f f1-1f f2-1f 
        f0-2f f1-2f f2-2f 
        f0-3f f1-3f f2-3f 
        f0-4f f1-4f f2-4f 
        f0-5f f1-5f f2-5f 
        f0-6f f1-6f f2-6f 
        f0-7f f1-7f f2-7f 
        f0-8f f1-8f f2-8f 
        shape0 shape1 
        key0-0 
)
(:init
(arm-empty)
(place f0-0f)
(place f1-0f)
(place f2-0f)
(place f0-1f)
(place f1-1f)
(place f2-1f)
(place f0-2f)
(place f1-2f)
(place f2-2f)
(place f0-3f)
(place f1-3f)
(place f2-3f)
(place f0-4f)
(place f1-4f)
(place f2-4f)
(place f0-5f)
(place f1-5f)
(place f2-5f)
(place f0-6f)
(place f1-6f)
(place f2-6f)
(place f0-7f)
(place f1-7f)
(place f2-7f)
(place f0-8f)
(place f1-8f)
(place f2-8f)
(shape shape0)
(shape shape1)
(key key0-0)
(key-shape key0-0 shape0)
(conn f0-0f f1-0f)
(conn f1-0f f2-0f)
(conn f0-1f f1-1f)
(conn f1-1f f2-1f)
(conn f0-2f f1-2f)
(conn f1-2f f2-2f)
(conn f0-3f f1-3f)
(conn f1-3f f2-3f)
(conn f0-4f f1-4f)
(conn f1-4f f2-4f)
(conn f0-5f f1-5f)
(conn f1-5f f2-5f)
(conn f0-6f f1-6f)
(conn f1-6f f2-6f)
(conn f0-7f f1-7f)
(conn f1-7f f2-7f)
(conn f0-8f f1-8f)
(conn f1-8f f2-8f)
(conn f0-0f f0-1f)
(conn f1-0f f1-1f)
(conn f2-0f f2-1f)
(conn f0-1f f0-2f)
(conn f1-1f f1-2f)
(conn f2-1f f2-2f)
(conn f0-2f f0-3f)
(conn f1-2f f1-3f)
(conn f2-2f f2-3f)
(conn f0-3f f0-4f)
(conn f1-3f f1-4f)
(conn f2-3f f2-4f)
(conn f0-4f f0-5f)
(conn f1-4f f1-5f)
(conn f2-4f f2-5f)
(conn f0-5f f0-6f)
(conn f1-5f f1-6f)
(conn f2-5f f2-6f)
(conn f0-6f f0-7f)
(conn f1-6f f1-7f)
(conn f2-6f f2-7f)
(conn f0-7f f0-8f)
(conn f1-7f f1-8f)
(conn f2-7f f2-8f)
(conn f1-0f f0-0f)
(conn f2-0f f1-0f)
(conn f1-1f f0-1f)
(conn f2-1f f1-1f)
(conn f1-2f f0-2f)
(conn f2-2f f1-2f)
(conn f1-3f f0-3f)
(conn f2-3f f1-3f)
(conn f1-4f f0-4f)
(conn f2-4f f1-4f)
(conn f1-5f f0-5f)
(conn f2-5f f1-5f)
(conn f1-6f f0-6f)
(conn f2-6f f1-6f)
(conn f1-7f f0-7f)
(conn f2-7f f1-7f)
(conn f1-8f f0-8f)
(conn f2-8f f1-8f)
(conn f0-1f f0-0f)
(conn f1-1f f1-0f)
(conn f2-1f f2-0f)
(conn f0-2f f0-1f)
(conn f1-2f f1-1f)
(conn f2-2f f2-1f)
(conn f0-3f f0-2f)
(conn f1-3f f1-2f)
(conn f2-3f f2-2f)
(conn f0-4f f0-3f)
(conn f1-4f f1-3f)
(conn f2-4f f2-3f)
(conn f0-5f f0-4f)
(conn f1-5f f1-4f)
(conn f2-5f f2-4f)
(conn f0-6f f0-5f)
(conn f1-6f f1-5f)
(conn f2-6f f2-5f)
(conn f0-7f f0-6f)
(conn f1-7f f1-6f)
(conn f2-7f f2-6f)
(conn f0-8f f0-7f)
(conn f1-8f f1-7f)
(conn f2-8f f2-7f)
(open f0-0f)
(open f1-0f)
(open f2-0f)
(open f0-1f)
(open f1-1f)
(open f2-1f)
(open f0-2f)
(open f1-2f)
(open f2-2f)
(open f0-3f)
(open f1-3f)
(open f2-3f)
(open f0-4f)
(open f1-4f)
(open f2-4f)
(open f0-5f)
(open f1-5f)
(open f2-5f)
(open f0-6f)
(open f1-6f)
(open f2-6f)
(open f0-7f)
(open f2-7f)
(open f0-8f)
(open f1-8f)
(open f2-8f)
(locked f1-7f)
(lock-shape f1-7f shape0)
(at key0-0 f0-7f)
(at-robot f0-6f)
)
(:goal
(and
(at key0-0 f1-1f)
)
)
)
