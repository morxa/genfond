

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 )
(:init
(arm-empty)
(on-table b1)
(on-table b2)
(on-table b3)
(on b4 b5)
(on b5 b2)
(clear b1)
(clear b3)
(clear b4)
)
(:goal
(and
(on b2 b4)
(on b4 b3))
)
)


