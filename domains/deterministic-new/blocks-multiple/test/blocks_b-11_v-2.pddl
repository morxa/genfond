

(define (problem BW-rand-11)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(arm-empty)
(on b1 b7)
(on-table b2)
(on b3 b9)
(on b4 b11)
(on-table b5)
(on-table b6)
(on b7 b3)
(on b8 b2)
(on b9 b10)
(on b10 b6)
(on b11 b5)
(clear b1)
(clear b4)
(clear b8)
)
(:goal
(and
(on b1 b6)
(on b2 b7)
(on b3 b2)
(on b5 b1)
(on b6 b11)
(on b8 b3)
(on b9 b4)
(on b10 b8)
(on b11 b9))
)
)


