


(define (problem logistics-c1-s2-p8-a2)
(:domain logistics-strips)
(:objects a0 a1 
          c0 
          t0 
          l0-0 l0-1 
          p0 p1 p2 p3 p4 p5 p6 p7 
)
(:init
    (AIRPLANE a0)
    (AIRPLANE a1)
    (CITY c0)
    (TRUCK t0)
    (LOCATION l0-0)
    (in-city  l0-0 c0)
    (LOCATION l0-1)
    (in-city  l0-1 c0)
    (AIRPORT l0-0)
    (PACKAGE p0)
    (PACKAGE p1)
    (PACKAGE p2)
    (PACKAGE p3)
    (PACKAGE p4)
    (PACKAGE p5)
    (PACKAGE p6)
    (PACKAGE p7)
    (at t0 l0-1)
    (at p0 l0-0)
    (at p1 l0-0)
    (at p2 l0-1)
    (at p3 l0-0)
    (at p4 l0-1)
    (at p5 l0-1)
    (at p6 l0-0)
    (at p7 l0-1)
    (at a0 l0-0)
    (at a1 l0-0)
)
(:goal
    (and
        (at p0 l0-0)
        (at p1 l0-0)
        (at p2 l0-1)
        (at p3 l0-0)
        (at p4 l0-1)
        (at p5 l0-1)
        (at p6 l0-1)
        (at p7 l0-1)
    )
)
)


