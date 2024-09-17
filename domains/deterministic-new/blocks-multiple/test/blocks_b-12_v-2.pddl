

(define (problem BW-rand-12)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b8)
(on b2 b12)
(on-table b3)
(on b4 b1)
(on b5 b9)
(on-table b6)
(on-table b7)
(on b8 b6)
(on b9 b2)
(on b10 b3)
(on b11 b10)
(on-table b12)
(clear b4)
(clear b5)
(clear b7)
(clear b11)
)
(:goal
(and
(on b1 b8)
(on b3 b7)
(on b4 b10)
(on b5 b9)
(on b7 b5)
(on b8 b12)
(on b9 b2)
(on b10 b11)
(on b12 b4))
)
)


