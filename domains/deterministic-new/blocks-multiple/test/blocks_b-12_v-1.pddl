

(define (problem BW-rand-12)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b10)
(on b2 b7)
(on-table b3)
(on b4 b1)
(on b5 b4)
(on b6 b11)
(on b7 b3)
(on b8 b6)
(on-table b9)
(on b10 b2)
(on b11 b5)
(on-table b12)
(clear b8)
(clear b9)
(clear b12)
)
(:goal
(and
(on b2 b8)
(on b3 b11)
(on b4 b7)
(on b5 b4)
(on b6 b1)
(on b7 b2)
(on b10 b3)
(on b11 b6)
(on b12 b9))
)
)


