(define (problem blocks-011-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b7) (clear b9) (handempty) (on b1 b0) (on b10 b5) (on b3 b4) (on b4 b6) (on b5 b2) (on b7 b10) (on b8 b3) (on b9 b8) (ontable b0) (ontable b2) (ontable b6))
    (:goal (on b0 b1))
)