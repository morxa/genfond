(define (problem blocks-010-08)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b7) (handempty) (on b0 b2) (on b1 b9) (on b3 b0) (on b4 b6) (on b5 b3) (on b6 b8) (on b7 b5) (on b9 b4) (ontable b2) (ontable b8))
    (:goal (on b0 b6))
)