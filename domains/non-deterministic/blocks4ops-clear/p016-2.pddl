(define (problem blocks-016-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b15 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b6) (handempty) (on b0 b12) (on b1 b13) (on b10 b9) (on b11 b2) (on b12 b11) (on b13 b3) (on b14 b10) (on b2 b14) (on b3 b15) (on b4 b0) (on b5 b4) (on b6 b5) (on b7 b1) (on b8 b7) (on b9 b8) (ontable b15))
    (:goal (clear b10))
)