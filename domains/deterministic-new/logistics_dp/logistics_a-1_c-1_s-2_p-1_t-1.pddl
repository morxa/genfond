


(define (problem logistics-c1-s2-p1-a1)
(:domain logistics-strips)
(:objects a0 
          c0 
          t0 
          l0-0 l0-1 
          p0 
)
(:init
    (AIRPLANE a0)
    (CITY c0)
    (TRUCK t0)
    (LOCATION l0-0)
    (in-city  l0-0 c0)
    (LOCATION l0-1)
    (in-city  l0-1 c0)
    (AIRPORT l0-0)
    (PACKAGE p0)
    (at t0 l0-1)
    (at p0 l0-0)
    (at a0 l0-0)
)
(:goal
    (and
        (at p0 l0-0)
    )
)
)


