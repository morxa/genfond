(define (problem problem-20-1)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b20 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b10) (clear b11) (clear b16) (clear b20) (clear b6) (on b1 b19) (on b12 b14) (on b13 b12) (on b14 b8) (on b16 b17) (on b17 b9) (on b18 b2) (on b19 b4) (on b20 b5) (on b3 b15) (on b4 b13) (on b5 b1) (on b6 b18) (on b7 b3) (on b9 b7) (on-table b10) (on-table b11) (on-table b15) (on-table b2) (on-table b8))
    (:goal (and (on b1 b7) (on b2 b5) (on b3 b6) (on b5 b20) (on b6 b11) (on b7 b13) (on b8 b4) (on b10 b3) (on b11 b8) (on b12 b10) (on b13 b15) (on b14 b17) (on b15 b18) (on b16 b9) (on b17 b19) (on b18 b16) (on b19 b2)))
)