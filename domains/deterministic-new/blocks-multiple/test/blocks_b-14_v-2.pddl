

(define (problem BW-rand-14)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 )
(:init
(arm-empty)
(on b1 b5)
(on b2 b6)
(on b3 b14)
(on b4 b13)
(on b5 b8)
(on-table b6)
(on b7 b10)
(on b8 b7)
(on-table b9)
(on b10 b2)
(on-table b11)
(on b12 b4)
(on b13 b1)
(on b14 b11)
(clear b3)
(clear b9)
(clear b12)
)
(:goal
(and
(on b1 b5)
(on b2 b12)
(on b3 b9)
(on b5 b6)
(on b6 b7)
(on b7 b14)
(on b8 b4)
(on b9 b11)
(on b10 b13)
(on b11 b2)
(on b12 b1)
(on b14 b10))
)
)


