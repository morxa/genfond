(define (problem blocks-007-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6)
    (:init (clear b2) (handempty) (on b1 b3) (on b2 b4) (on b3 b6) (on b4 b1) (on b5 b0) (on b6 b5) (ontable b0))
    (:goal (on b0 b6))
)