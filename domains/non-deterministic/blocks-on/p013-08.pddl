(define (problem blocks-013-08)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b3) (clear b7) (handempty) (on b0 b10) (on b1 b12) (on b11 b1) (on b12 b2) (on b2 b0) (on b3 b9) (on b5 b6) (on b6 b8) (on b7 b5) (on b8 b11) (on b9 b4) (ontable b10) (ontable b4))
    (:goal (on b9 b3))
)