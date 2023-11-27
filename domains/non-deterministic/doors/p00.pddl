(define (problem p00)
(:domain doors)
(:objects
L1 - location
L2 - location
D2 - door
)
(:init
(player-at L1)
(initial-location L1)
(open D2)
(door-in D2 L2)
(door-out D2 L1)
(final-location L2)
)
(:goal (player-at L2)))
