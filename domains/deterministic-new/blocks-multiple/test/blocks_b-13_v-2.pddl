

(define (problem BW-rand-13)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(arm-empty)
(on b1 b10)
(on-table b2)
(on b3 b12)
(on b4 b1)
(on-table b5)
(on b6 b3)
(on b7 b13)
(on b8 b11)
(on-table b9)
(on b10 b8)
(on b11 b2)
(on b12 b9)
(on b13 b4)
(clear b5)
(clear b6)
(clear b7)
)
(:goal
(and
(on b1 b6)
(on b2 b1)
(on b3 b7)
(on b4 b5)
(on b5 b12)
(on b7 b4)
(on b8 b3)
(on b9 b13)
(on b10 b8)
(on b11 b10))
)
)


