(define (problem blocks-005-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b3) (handempty) (on b1 b4) (on b2 b1) (on b3 b2) (on b4 b0) (ontable b0))
    (:goal (clear b0))
)