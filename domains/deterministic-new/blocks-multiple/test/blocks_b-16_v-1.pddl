

(define (problem BW-rand-16)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 )
(:init
(arm-empty)
(on b1 b5)
(on-table b2)
(on b3 b16)
(on b4 b7)
(on b5 b6)
(on b6 b13)
(on b7 b8)
(on b8 b9)
(on b9 b10)
(on b10 b1)
(on b11 b12)
(on b12 b14)
(on b13 b3)
(on b14 b4)
(on b15 b2)
(on-table b16)
(clear b11)
(clear b15)
)
(:goal
(and
(on b1 b2)
(on b2 b12)
(on b3 b10)
(on b4 b11)
(on b6 b15)
(on b7 b3)
(on b8 b13)
(on b9 b7)
(on b12 b16)
(on b13 b1)
(on b15 b9)
(on b16 b14))
)
)


