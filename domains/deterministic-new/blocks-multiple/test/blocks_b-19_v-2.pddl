

(define (problem BW-rand-19)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 )
(:init
(arm-empty)
(on b1 b8)
(on b2 b14)
(on b3 b18)
(on b4 b11)
(on-table b5)
(on b6 b5)
(on b7 b10)
(on b8 b4)
(on-table b9)
(on b10 b12)
(on b11 b6)
(on b12 b19)
(on b13 b15)
(on b14 b17)
(on-table b15)
(on b16 b3)
(on-table b17)
(on b18 b13)
(on-table b19)
(clear b1)
(clear b2)
(clear b7)
(clear b9)
(clear b16)
)
(:goal
(and
(on b1 b12)
(on b2 b11)
(on b3 b9)
(on b4 b10)
(on b5 b18)
(on b8 b15)
(on b9 b6)
(on b10 b14)
(on b12 b8)
(on b14 b17)
(on b15 b13)
(on b16 b1)
(on b17 b3)
(on b19 b5))
)
)


