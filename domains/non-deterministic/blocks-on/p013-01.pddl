(define (problem blocks-013-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b2) (clear b3) (clear b8) (handempty) (on b1 b6) (on b11 b9) (on b12 b5) (on b2 b7) (on b3 b10) (on b4 b11) (on b5 b4) (on b7 b1) (on b8 b12) (on b9 b0) (ontable b0) (ontable b10) (ontable b6))
    (:goal (on b0 b4))
)