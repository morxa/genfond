

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 )
(:init
(arm-empty)
(on b1 b10)
(on b2 b3)
(on-table b3)
(on-table b4)
(on-table b5)
(on-table b6)
(on-table b7)
(on b8 b5)
(on-table b9)
(on b10 b7)
(clear b1)
(clear b2)
(clear b4)
(clear b6)
(clear b8)
(clear b9)
)
(:goal
(and
(on b1 b4)
(on b2 b8)
(on b3 b1)
(on b4 b9)
(on b5 b2)
(on b7 b6)
(on b8 b10)
(on b9 b7))
)
)


